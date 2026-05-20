/**
 * @file ReceiverIO.cpp
 * @brief ReceiverArray file I/O methods (ParseTextFile, SaveTo*, GatherData)
 */

#include "srcrecv/Receiver.hpp"
#include "srcrecv/HDF5IOSchema.hpp"
#include "srcrecv/HDF5SourceReceiverWriter.hpp"
#include "srcrecv/SUFormat.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cstring>
#include <cmath>
#include <hdf5.h>

namespace SEM {

// =============================================================================
// ReceiverArray - File I/O
// =============================================================================

void ReceiverArray::ParseTextFile(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        MFEM_ABORT("Unable to open receiver file: " << filepath);
    }

    std::string line;
    int num_receivers;

    // Read number of receivers
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> num_receivers;

    // Parse each receiver
    for (int i = 0; i < num_receivers; i++) {
        std::getline(file, line);
        std::istringstream ls(line);

        int type_int;
        std::string name;
        real_t x, y, z = 0.0;
        real_t weight = 1.0;

        if (space_dim_ == 2) {
            ls >> type_int >> name >> x >> y >> weight;
        } else {
            ls >> type_int >> name >> x >> y >> z;
        }

        ReceiverType type = IntToReceiverType(type_int);

        // Check domain type compatibility - filter incompatible types
        bool is_solid_receiver = (type_int < 10);
        if (domain_ == DomainType::Solid && !is_solid_receiver) {
            // Log filtered receiver (acoustic type in elastic simulation)
            filtered_log_.push_back({name, ReceiverTypeToString(type),
                "acoustic receiver type in elastic simulation"});
            continue;  // Skip this receiver
        }
        if (domain_ == DomainType::Fluid && is_solid_receiver) {
            // Log filtered receiver (elastic type in acoustic simulation)
            filtered_log_.push_back({name, ReceiverTypeToString(type),
                "elastic receiver type in acoustic simulation"});
            continue;  // Skip this receiver
        }

        // Determine number of components
        int ncomp = 1;
        if (type == ReceiverType::Displacement ||
            type == ReceiverType::Velocity ||
            type == ReceiverType::Acceleration) {
            ncomp = space_dim_;
        } else if (type == ReceiverType::Gradient) {
            ncomp = space_dim_ * space_dim_;
        }

        // Create position vector
        Vector pos(space_dim_);
        pos[0] = x;
        pos[1] = y;
        if (space_dim_ == 3) {
            pos[2] = z;
        }

        // Create receiver
        ReceiverData rec(name, pos, type, nt_, dt_, ncomp);
        rec.SetWeight(weight);

        receivers_[type].push_back(std::move(rec));
    }

    file.close();
}

void ReceiverArray::SetOutputConfig(const std::vector<std::string>& formats,
                                    const std::string& outdir,
                                    const std::string& filename) {
    output_formats_ = formats;
    output_dir_ = outdir;
    output_filename_ = filename;
}

void ReceiverArray::SetOutputConfig(const std::string& format,
                                    const std::string& outdir,
                                    const std::string& filename) {
    SetOutputConfig(std::vector<std::string>{format}, outdir, filename);
}

void ReceiverArray::SetOutputSourceContext(
    std::vector<HDF5SourceWriteEntry> sources) {
    output_sources_ = std::move(sources);
}

void ReceiverArray::Save(int source_id, real_t t0,
                         const Vector* source_pos) {
    // Hoist GPU→host flush and MPI gather out of each format-specific helper
    // so they run at most once per Save() call, regardless of how many
    // formats are requested.
    FlushDeviceBuffer();

    bool needs_gather = false;
    for (const auto& f : output_formats_) {
        if (f == "hdf5" || f == "su") { needs_gather = true; break; }
    }
    if (needs_gather) {
        GatherData();
    }

    for (const auto& fmt : output_formats_) {
        if (fmt == "ascii") {
            SaveToTextFiles(output_dir_, source_id, t0);
        } else if (fmt == "hdf5") {
            SaveToHDF5(output_dir_, output_filename_, source_id, true, t0);
        } else if (fmt == "su") {
            SaveToSU(output_dir_, output_filename_, source_id, t0, source_pos);
        }
    }
}

void ReceiverArray::SaveToTextFiles(const std::string& outdir,
                                    int source_id,
                                    real_t t0) {
    // Precondition: Save() has already called FlushDeviceBuffer().
    for (auto& entry : receivers_) {
        for (auto& rec : entry.second) {
            if (local_rank_ != rec.OwnerRank()) continue;

            int ncomp = rec.NumComponents();
            for (int c = 0; c < ncomp; c++) {
                std::string filename = GetSingleTraceName(outdir, rec, c, source_id);
                std::ofstream ofile(filename);
                if (!ofile) {
                    MFEM_ABORT("Cannot open file: " << filename);
                }
                // Full round-trip precision for real_t (otherwise the default
                // 6-digit stream precision silently drops ~10 digits).
                ofile << std::setprecision(
                            std::numeric_limits<real_t>::max_digits10);

                for (int it = 0; it < nt_; it++) {
                    ofile << t0 + it * dt_ << " " << rec.Data()(it, c) << "\n";
                }
                ofile.close();
            }
        }
    }
}

std::string ReceiverArray::GetSingleTraceName(const std::string& outdir,
                                               const ReceiverData& rec,
                                               int comp, int source_id) const {
    std::string comp_name;
    std::string prefix;

    ReceiverType type = rec.Type();

    if (type == ReceiverType::Displacement ||
        type == ReceiverType::Velocity ||
        type == ReceiverType::Acceleration) {
        if (space_dim_ == 2) {
            comp_name = (comp == 0) ? "x" : "y";
        } else {
            const char* names[] = {"x", "y", "z"};
            comp_name = names[comp];
        }
    }

    switch (type) {
        case ReceiverType::Displacement: prefix = "d"; break;
        case ReceiverType::Velocity: prefix = "v"; break;
        case ReceiverType::Acceleration: prefix = "a"; break;
        case ReceiverType::Pressure: prefix = "p"; break;
        case ReceiverType::Gradient: prefix = "g"; break;
    }

    std::string id_str = MakeSourceIdString(source_id);

    if (type == ReceiverType::Pressure) {
        return outdir + "/" + rec.Name() + "_" + id_str + "." + prefix;
    } else {
        return outdir + "/" + rec.Name() + "_" + comp_name + "_" + id_str + "." + prefix;
    }
}

void ReceiverArray::SaveToHDF5(const std::string& outdir,
                               const std::string& filename,
                               int source_id,
                               bool overwrite,
                               real_t t0) {
    // v2.0 schema (see include/srcrecv/HDF5IOSchema.hpp). One per-shot
    // file holds a single shot under `/shots/0000/`; the driver merge
    // tool reorganises multiple per-shot files under
    // `/shots/0000`, `/shots/0001`, ...
    //
    //   /                              attrs: format_version="2.0", dt, t0,
    //                                        n_samples, space_dim,
    //                                        coord_system="cartesian",
    //                                        units="SI"
    //   /shots/0000/                   attrs: shot_id=0
    //       /receivers/<name>/         attrs: position
    //           /pressure              (nt,)              for PS
    //           /displacement          (space_dim, nt)    for DISP
    //           /velocity, /acceleration                  likewise
    //   GRAD receivers are skipped here (no observed channel defined).
    //
    // Precondition: Save() has already called FlushDeviceBuffer() and GatherData().
    std::string base_filename = filename;
    if (base_filename.size() > 3 &&
        base_filename.substr(base_filename.size() - 3) == ".h5") {
        base_filename = base_filename.substr(0, base_filename.size() - 3);
    }
    std::string full_filename = outdir + "/" + base_filename
                                + MakeSourceIdString(source_id) + ".h5";

    if (local_rank_ != 0) return;

    auto write_scalar_attr = [](hid_t owner, const char* name, hid_t type,
                                const void* v) {
        hid_t sp = H5Screate(H5S_SCALAR);
        hid_t a = H5Acreate2(owner, name, type, sp, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(a, type, v);
        H5Aclose(a);
        H5Sclose(sp);
    };
    auto write_vector_attr = [](hid_t owner, const char* name, hid_t type,
                                hsize_t n, const void* v) {
        hsize_t d[1] = {n};
        hid_t sp = H5Screate_simple(1, d, nullptr);
        hid_t a = H5Acreate2(owner, name, type, sp, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(a, type, v);
        H5Aclose(a);
        H5Sclose(sp);
    };
    auto write_str_attr = [](hid_t owner, const char* name, const char* value) {
        hid_t t = H5Tcopy(H5T_C_S1);
        H5Tset_size(t, H5T_VARIABLE);
        H5Tset_cset(t, H5T_CSET_UTF8);
        hid_t sp = H5Screate(H5S_SCALAR);
        hid_t a = H5Acreate2(owner, name, t, sp, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(a, t, &value);
        H5Aclose(a);
        H5Sclose(sp);
        H5Tclose(t);
    };

    hid_t file = H5Fcreate(full_filename.c_str(),
                           overwrite ? H5F_ACC_TRUNC : H5F_ACC_EXCL,
                           H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0 && !overwrite) {
        file = H5Fopen(full_filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    }
    MFEM_VERIFY(file >= 0, "Cannot open HDF5 file: " << full_filename);

    // ---- Root attributes (v2.0) ----
    write_str_attr(file, HDF5Schema::kAttrFormatVersion,
                   HDF5Schema::kFormatVersion);
    double dt_d = static_cast<double>(dt_);
    double t0_d = static_cast<double>(t0);
    write_scalar_attr(file, HDF5Schema::kAttrDt, H5T_NATIVE_DOUBLE, &dt_d);
    write_scalar_attr(file, HDF5Schema::kAttrT0, H5T_NATIVE_DOUBLE, &t0_d);
    int64_t n_samples_v = nt_;
    write_scalar_attr(file, HDF5Schema::kAttrNSamples, H5T_NATIVE_INT64,
                      &n_samples_v);
    int32_t space_dim_v = space_dim_;
    write_scalar_attr(file, HDF5Schema::kAttrSpaceDim, H5T_NATIVE_INT32,
                      &space_dim_v);
    write_str_attr(file, HDF5Schema::kAttrCoordSystem,
                   HDF5Schema::kDefaultCoordSystem);
    write_str_attr(file, HDF5Schema::kAttrUnits, HDF5Schema::kDefaultUnits);

    // ---- /shots/<NNNN>/ (single-shot per-file output) ----
    // Each per-shot file writes its source's shot_id (set by SetShotId()
    // from the input config). For YAML-inline runs without sources.shot_id
    // this defaults to 0, matching the legacy behavior.
    hid_t shots_root = H5Gcreate2(file, HDF5Schema::kGroupShots,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const std::string shot_key = HDF5Schema::ShotKey(shot_id_);
    hid_t shot_group = H5Gcreate2(shots_root, shot_key.c_str(),
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    int32_t shot_id_v = shot_id_;
    write_scalar_attr(shot_group, HDF5Schema::kAttrShotId, H5T_NATIVE_INT32,
                      &shot_id_v);

    // ---- /shots/0000/sources/ (Stage 5; self-roundtrip) ----
    // Only written when the simulation has registered descriptors via
    // SetOutputSourceContext. Receiver-only callers (legacy) get the
    // pre-Stage-5 layout with no /sources/ subgroup.
    if (!output_sources_.empty()) {
        WriteSourcesIntoShotGroup(shot_group, space_dim_, output_sources_);
    }

    // ---- Pivot (type,name)-keyed layout into name-keyed layout ----
    // The internal map is `receivers_[type] -> vector<ReceiverData>`, but
    // the v2.0 schema is receiver-first. Build a side index from
    // receiver name → list of (type, pointer) so each receiver's
    // channels are emitted under one shared group.
    std::vector<std::string> order_names;  // preserve first-seen order
    std::map<std::string, std::vector<ReceiverData*>> by_name;
    for (auto& entry : receivers_) {
        for (auto& rec : entry.second) {
            auto it = by_name.find(rec.Name());
            if (it == by_name.end()) {
                order_names.push_back(rec.Name());
                by_name[rec.Name()] = {&rec};
            } else {
                it->second.push_back(&rec);
            }
        }
    }

    hid_t receivers_group = H5Gcreate2(shot_group, HDF5Schema::kGroupReceivers,
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    for (const auto& name : order_names) {
        auto& recs = by_name[name];
        hid_t rg = H5Gcreate2(receivers_group, name.c_str(),
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        // position attribute (double array, size=space_dim)
        std::vector<double> pos_d(space_dim_);
        const Vector& pos = recs.front()->Position();
        for (int d = 0; d < space_dim_; ++d) {
            pos_d[d] = static_cast<double>(pos[d]);
        }
        write_vector_attr(rg, HDF5Schema::kAttrPosition, H5T_NATIVE_DOUBLE,
                          static_cast<hsize_t>(space_dim_), pos_d.data());

        // Emit one dataset per (type, receiver). Observed schema uses
        // float32 regardless of real_t; cast here.
        for (ReceiverData* rec : recs) {
            const ReceiverType type = rec->Type();
            const std::string channel = ReceiverTypeToObservedChannel(type);
            if (channel.empty()) {
                MFEM_WARNING("HDF5 observed schema has no channel for "
                             "receiver type '" << ReceiverTypeToString(type)
                             << "' (receiver '" << rec->Name()
                             << "'); skipping.");
                continue;
            }

            const int ncomp = rec->NumComponents();
            const int nt    = rec->NumTimeSteps();
            // DenseMatrix `data_` is [nt × ncomp] column-major — memory
            // is comp0[0..nt-1] || comp1[0..nt-1] || ... which equals
            // HDF5 row-major shape (ncomp, nt). No transpose required.
            const real_t* src = rec->Data().GetData();

            std::vector<float> buf(static_cast<size_t>(ncomp) * nt);
            for (size_t i = 0; i < buf.size(); ++i) {
                buf[i] = static_cast<float>(src[i]);
            }

            // Scalar channels (pressure): shape (nt,) — observed reader
            // rejects a 2-D pressure.
            hid_t sp;
            if (ncomp == 1) {
                hsize_t d1[1] = {static_cast<hsize_t>(nt)};
                sp = H5Screate_simple(1, d1, nullptr);
            } else {
                hsize_t d2[2] = {static_cast<hsize_t>(ncomp),
                                 static_cast<hsize_t>(nt)};
                sp = H5Screate_simple(2, d2, nullptr);
            }
            hid_t ds = H5Dcreate2(rg, channel.c_str(),
                                  H5T_NATIVE_FLOAT, sp,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     buf.data());
            H5Dclose(ds);
            H5Sclose(sp);
        }

        H5Gclose(rg);
    }

    H5Gclose(receivers_group);
    H5Gclose(shot_group);
    H5Gclose(shots_root);
    H5Fclose(file);
}

// =============================================================================
// SU Format Output
// =============================================================================

std::string ReceiverArray::GetSUFileName(const std::string& outdir,
                                          const std::string& filename,
                                          ReceiverType type,
                                          int comp,
                                          int source_id) const {
    std::string type_str;
    switch (type) {
        case ReceiverType::Displacement: type_str = "DISP"; break;
        case ReceiverType::Velocity:     type_str = "VEL"; break;
        case ReceiverType::Acceleration: type_str = "ACC"; break;
        case ReceiverType::Pressure:     type_str = "PS"; break;
        case ReceiverType::Gradient:     type_str = "GRAD"; break;
    }

    std::string comp_str;
    if (type == ReceiverType::Pressure) {
        // Scalar: no component suffix
    } else if (type == ReceiverType::Displacement ||
               type == ReceiverType::Velocity ||
               type == ReceiverType::Acceleration) {
        const char* names[] = {"x", "y", "z"};
        comp_str = std::string("_") + names[comp];
    } else if (type == ReceiverType::Gradient) {
        const char* dim_names[] = {"x", "y", "z"};
        int row = comp / space_dim_;
        int col_idx = comp % space_dim_;
        comp_str = std::string("_") + dim_names[row] + dim_names[col_idx];
    }

    std::string id_str = MakeSourceIdString(source_id);
    return outdir + "/" + filename + "_" + type_str + comp_str + "_" + id_str + ".su";
}

void ReceiverArray::SaveToSU(const std::string& outdir,
                              const std::string& filename,
                              int source_id,
                              real_t t0,
                              const Vector* source_pos) {
    // Precondition: Save() has already called FlushDeviceBuffer() and GatherData().
    // Only rank 0 writes
    if (local_rank_ != 0) return;

    // Warn if nt exceeds uint16_t range
    if (nt_ > 65535) {
        MFEM_WARNING("SU format: nt=" << nt_
                     << " exceeds uint16_t max (65535). "
                     << "ns header field will be truncated.");
    }

    for (auto& entry : receivers_) {
        ReceiverType type = entry.first;
        auto& receivers = entry.second;
        if (receivers.empty()) continue;

        int ncomp = receivers[0].NumComponents();

        for (int c = 0; c < ncomp; c++) {
            std::string su_filename = GetSUFileName(outdir, filename,
                                                     type, c, source_id);
            std::ofstream ofile(su_filename, std::ios::binary);
            if (!ofile) {
                MFEM_ABORT("Cannot open SU file: " << su_filename);
            }

            int num_traces = static_cast<int>(receivers.size());
            std::vector<float> trace_buf(nt_);

            for (int tr = 0; tr < num_traces; tr++) {
                auto& rec = receivers[tr];

                // Build SU trace header
                SUTraceHeader hdr;
                std::memset(&hdr, 0, sizeof(hdr));

                hdr.tracl = tr + 1;
                hdr.tracr = tr + 1;
                hdr.fldr  = source_id + 1;
                hdr.tracf = tr + 1;
                hdr.ep    = source_id + 1;
                hdr.trid  = 1;  // seismic data
                hdr.ns    = static_cast<uint16_t>(nt_);
                hdr.dt    = static_cast<uint16_t>(dt_ * 1.0e6);

                // Coordinates (scalco = -1000 for mm precision)
                hdr.scalco = -1000;
                hdr.counit = 1;  // meters
                const Vector& pos = rec.Position();
                hdr.gx = static_cast<int32_t>(pos[0] * 1000.0);
                if (space_dim_ >= 2) {
                    hdr.gy = static_cast<int32_t>(pos[1] * 1000.0);
                }

                // Source coordinates and offset
                if (source_pos) {
                    hdr.sx = static_cast<int32_t>((*source_pos)[0] * 1000.0);
                    if (space_dim_ >= 2) {
                        hdr.sy = static_cast<int32_t>((*source_pos)[1] * 1000.0);
                    }
                    // Compute offset (source-receiver distance)
                    real_t dist2 = 0.0;
                    //in 2D, compute using x only
                    if (space_dim_ == 2) {
                        real_t diff = pos[0] - (*source_pos)[0];
                        dist2 = diff * diff;
                    } 
                    else {
                        for (int d = 0; d < space_dim_; d++) {
                            real_t diff = pos[d] - (*source_pos)[d];
                            dist2 += diff * diff;
                        }
                    }
                    hdr.offset = static_cast<int32_t>(std::sqrt(dist2));
                }

                // CWP local fields
                hdr.d1  = static_cast<float>(dt_);
                hdr.f1  = static_cast<float>(t0);
                hdr.ntr = static_cast<int32_t>(num_traces);

                // Write header (240 bytes)
                ofile.write(reinterpret_cast<const char*>(&hdr),
                            sizeof(SUTraceHeader));

                // Write data as float32
                // DenseMatrix data_ is [nt x ncomp] column-major:
                // column c starts at GetData() + c * nt_
                const real_t* col = rec.Data().GetData() + c * nt_;
                for (int i = 0; i < nt_; i++) {
                    trace_buf[i] = static_cast<float>(col[i]);
                }

                ofile.write(reinterpret_cast<const char*>(trace_buf.data()),
                            nt_ * sizeof(float));
            }
            ofile.close();
        }
    }
}

void ReceiverArray::GatherData() {
    // ID-based gather: only local receivers send their data to rank 0
    // This avoids allocating data storage for non-local receivers

    for (auto& entry : receivers_) {
        auto& receivers = entry.second;

        if (receivers.empty()) continue;

        int ncomp = receivers[0].NumComponents();
        int data_per_recv = nt_ * ncomp;

        // Collect local receiver IDs and data
        std::vector<int> local_ids;
        std::vector<real_t> local_data;

        for (const auto& rec : receivers) {
            if (rec.IsLocal()) {
                local_ids.push_back(rec.Id());
                const real_t* ptr = rec.Data().GetData();
                local_data.insert(local_data.end(), ptr, ptr + data_per_recv);
            }
        }

        int num_local_type = static_cast<int>(local_ids.size());

        // Gather counts from all ranks
        std::vector<int> all_counts(num_procs_);
        MPI_Gather(&num_local_type, 1, MPI_INT,
                   all_counts.data(), 1, MPI_INT, 0, *comm_);

        // Prepare displacement arrays for Gatherv
        std::vector<int> id_displs(num_procs_ + 1, 0);
        std::vector<int> data_displs(num_procs_ + 1, 0);

        if (local_rank_ == 0) {
            for (int r = 0; r < num_procs_; r++) {
                id_displs[r + 1] = id_displs[r] + all_counts[r];
                data_displs[r + 1] = data_displs[r] + all_counts[r] * data_per_recv;
            }
        }

        int total_recv = (local_rank_ == 0) ? id_displs[num_procs_] : 0;

        std::vector<int> all_ids(total_recv);
        std::vector<real_t> all_data(static_cast<size_t>(total_recv) * data_per_recv);

        // Gather IDs
        std::vector<int> id_counts = all_counts;
        MPI_Gatherv(local_ids.data(), num_local_type, MPI_INT,
                    all_ids.data(), id_counts.data(), id_displs.data(), MPI_INT,
                    0, *comm_);

        // Gather data
        std::vector<int> data_counts(num_procs_);
        for (int r = 0; r < num_procs_; r++) {
            data_counts[r] = all_counts[r] * data_per_recv;
        }
        MPI_Gatherv(local_data.data(), num_local_type * data_per_recv,
                    MPITypeMap<real_t>::mpi_type,
                    all_data.data(), data_counts.data(), data_displs.data(),
                    MPITypeMap<real_t>::mpi_type, 0, *comm_);

        // Rank 0: Store gathered data into receivers by ID
        if (local_rank_ == 0) {
            // Build ID -> receiver pointer map
            std::map<int, ReceiverData*> id_to_recv;
            for (auto& rec : receivers) {
                id_to_recv[rec.Id()] = &rec;
            }

            // Allocate and copy data
            for (int i = 0; i < total_recv; i++) {
                int id = all_ids[i];
                auto it = id_to_recv.find(id);
                if (it != id_to_recv.end()) {
                    ReceiverData* rec = it->second;
                    if (!rec->HasDataStorage()) {
                        rec->AllocateDataStorage();
                    }
                    std::memcpy(rec->Data().GetData(),
                                &all_data[static_cast<size_t>(i) * data_per_recv],
                                data_per_recv * sizeof(real_t));
                }
            }
        }
    }
}

void ReceiverArray::SaveFilteredLog(const std::string& filepath) const {
    // Only rank 0 writes the log
    if (local_rank_ != 0) return;

    if (filtered_log_.empty()) return;

    std::ofstream file(filepath);
    if (!file.is_open()) {
        MFEM_WARNING("Unable to open filtered receivers log file: " << filepath);
        return;
    }

    file << "# Filtered Receivers Log\n";
    file << "# Receivers excluded by domain type mismatch or because their\n";
    file << "# position lies outside the mesh / submesh.\n";
    file << "# Format: name, type, reason\n";
    file << "#\n";
    file << "# Total filtered: " << filtered_log_.size() << "\n";
    file << "#\n";

    for (const auto& entry : filtered_log_) {
        file << entry.name << ", " << entry.type << ", " << entry.reason << "\n";
    }

    file.close();
}

}  // namespace SEM
