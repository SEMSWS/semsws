/**
 * @file ReceiverRecord.cpp
 * @brief ReceiverArray recording methods (CPU and GPU)
 */

#include "srcrecv/Receiver.hpp"
#include <mfem.hpp>
#include "util/Profiler.hpp"

namespace SEM {

// =============================================================================
// ReceiverArray - Recording
// =============================================================================

void ReceiverArray::SetFields(ParGridFunction* u,
                              ParGridFunction* udot,
                              ParGridFunction* udot2) {
    u_ = u;
    udot_ = udot;
    udot2_ = udot2;
}

void ReceiverArray::Record(int step, int buffer_steps) {
    PROFILE_REGION("ReceiverRecord");

    if (step > nt_) {
        MFEM_ABORT("Time step exceeds nt: " << step << " > " << nt_);
    }

    // Use GPU path if device is enabled
    if (Device::Allows(Backend::DEVICE_MASK)) {
        RecordDevice(step, buffer_steps);
    } else {
        // CPU path
        if (space_dim_ == 2) {
            Record2D(step);
        } else {
            Record3D(step);
        }
    }
}

void ReceiverArray::Record2D(int step) {
    for (auto& entry : receivers_) {
        for (auto& rec : entry.second) {
            if (rec.IsLocal()) {
                RecordSingleReceiver2D(rec, step);
            }
        }
    }
}

void ReceiverArray::Record3D(int step) {
    for (auto& entry : receivers_) {
        for (auto& rec : entry.second) {
            if (rec.IsLocal()) {
                RecordSingleReceiver3D(rec, step);
            }
        }
    }
}

void ReceiverArray::RecordSingleReceiver2D(ReceiverData& rec, int step) {
    ReceiverType type = rec.Type();

    // Get field to sample
    ParGridFunction* field = nullptr;
    if (type == ReceiverType::Displacement) {
        field = u_;
    } else if (type == ReceiverType::Velocity) {
        field = udot_;
    } else if (type == ReceiverType::Acceleration || type == ReceiverType::Pressure) {
        field = udot2_;
    }

    if (!field) return;

    // Use cached interpolation data (computed once during setup)
    const Vector& shape = rec.CachedShape();
    const Array<int>& vdofs = rec.CachedVDofs();
    int ndof = shape.Size();

    Vector loc_data;
    loc_data.UseDevice(true);
    field->GetSubVector(vdofs, loc_data);

    int ncomp = rec.NumComponents();
    Vector sample(ncomp);
    sample = 0.0;

    const real_t* loc_ptr = loc_data.HostRead();
    const real_t* shape_ptr = shape.HostRead();

    if (type == ReceiverType::Pressure) {
        for (int k = 0; k < ndof; k++) {
            sample[0] -= loc_ptr[k] * shape_ptr[k];
        }
    } else {
        for (int c = 0; c < ncomp; c++) {
            for (int k = 0; k < ndof; k++) {
                sample[c] += loc_ptr[ndof * c + k] * shape_ptr[k];
            }
        }
    }

    rec.AddSample(step, sample);
}

void ReceiverArray::RecordSingleReceiver3D(ReceiverData& rec, int step) {
    ReceiverType type = rec.Type();

    ParGridFunction* field = nullptr;
    if (type == ReceiverType::Displacement) {
        field = u_;
    } else if (type == ReceiverType::Velocity) {
        field = udot_;
    } else if (type == ReceiverType::Acceleration || type == ReceiverType::Pressure) {
        field = udot2_;
    }

    if (!field) return;

    const Vector& shape = rec.CachedShape();
    const Array<int>& vdofs = rec.CachedVDofs();
    int ndof = shape.Size();

    Vector loc_data;
    loc_data.UseDevice(true);
    field->GetSubVector(vdofs, loc_data);

    int ncomp = rec.NumComponents();
    Vector sample(ncomp);
    sample = 0.0;

    const real_t* loc_ptr = loc_data.HostRead();
    const real_t* shape_ptr = shape.HostRead();

    if (type == ReceiverType::Pressure) {
        for (int k = 0; k < ndof; k++) {
            sample[0] -= loc_ptr[k] * shape_ptr[k];
        }
    } else {
        for (int c = 0; c < ncomp; c++) {
            for (int k = 0; k < ndof; k++) {
                sample[c] += loc_ptr[ndof * c + k] * shape_ptr[k];
            }
        }
    }

    rec.AddSample(step, sample);
}

// =============================================================================
// GPU Recording Implementation
// =============================================================================

void ReceiverArray::InitDeviceRecording(int max_buffer_steps) {
    // Initialize per ReceiverType
    for (auto& [type, receivers] : receivers_) {
        DeviceReceiverData& dd = device_data_[type];
        if (dd.initialized) continue;

        // Count local receivers and total DOFs
        int num_local = 0;
        int total_ndof = 0;
        for (const auto& rec : receivers) {
            if (rec.IsLocal()) {
                num_local++;
                total_ndof += rec.CachedShape().Size();
            }
        }

        if (num_local == 0) {
            dd.initialized = true;
            continue;
        }

        dd.num_local = num_local;
        dd.total_ndof = total_ndof;

        // ncomp depends on ReceiverType
        dd.ncomp = (type == ReceiverType::Pressure) ? 1 : space_dim_;

        // Buffer size: 0 means all steps
        dd.buffer_steps = (max_buffer_steps <= 0 || max_buffer_steps > nt_)
                          ? nt_ : max_buffer_steps;

        // Allocate device arrays
        dd.d_shape.SetSize(total_ndof);
        dd.d_vdofs.SetSize(total_ndof * dd.ncomp);
        dd.d_offsets.SetSize(num_local + 1);
        dd.d_buffer.SetSize(num_local * dd.buffer_steps * dd.ncomp);

        // Enable device memory
        dd.d_shape.UseDevice(true);
        dd.d_vdofs.GetMemory().UseDevice(true);
        dd.d_offsets.GetMemory().UseDevice(true);
        dd.d_buffer.UseDevice(true);

        // Pack shape/vdofs/offsets on host
        real_t* h_shape = dd.d_shape.HostWrite();
        int* h_vdofs = dd.d_vdofs.HostWrite();
        int* h_offsets = dd.d_offsets.HostWrite();

        int recv_idx = 0;
        int shape_offset = 0;
        h_offsets[0] = 0;

        for (const auto& rec : receivers) {
            if (!rec.IsLocal()) continue;

            const Vector& shape = rec.CachedShape();
            const Array<int>& vdofs = rec.CachedVDofs();
            int ndof = shape.Size();

            // Copy shape functions
            const real_t* shape_ptr = shape.HostRead();
            for (int k = 0; k < ndof; k++) {
                h_shape[shape_offset + k] = shape_ptr[k];
            }

            // Copy vdofs (already organized by component)
            const int* vdofs_ptr = vdofs.HostRead();
            for (int i = 0; i < ndof * dd.ncomp; i++) {
                h_vdofs[shape_offset * dd.ncomp + i] = vdofs_ptr[i];
            }

            shape_offset += ndof;
            recv_idx++;
            h_offsets[recv_idx] = shape_offset;
        }

        // Initialize buffer to zero
        dd.d_buffer = 0.0;
        dd.current_pos = 0;
        dd.start_step = 0;
        dd.initialized = true;

        // Sync to device (HostWrite wrote to host, now transfer to device)
        dd.d_shape.Read();
        dd.d_vdofs.Read();
        dd.d_offsets.Read();

        // Log buffer allocation
        size_t buffer_bytes = num_local * dd.buffer_steps * dd.ncomp * sizeof(real_t);
        mfem::out << "Receiver GPU buffer [" << ReceiverTypeToString(type) << "]: "
                  << num_local << " receivers, " << dd.ncomp << " comp, "
                  << dd.buffer_steps << " steps/flush, "
                  << (buffer_bytes / (1024.0 * 1024.0)) << " MB\n";
    }
}

void ReceiverArray::RecordDevice(int step, int max_buffer_steps) {
    // No receivers to record - early return
    if (receivers_.empty()) {
        return;
    }

    // Require explicit initialization via DeviceInit() if there are receivers
    MFEM_VERIFY(!device_data_.empty(),
                "RecordDevice() called before DeviceInit(). "
                "Call DeviceInit(max_buffer_steps) first.");

    // Record for each ReceiverType
    for (auto& [type, dd] : device_data_) {
        if (dd.num_local == 0) continue;

        // Flush if buffer is full
        if (dd.current_pos >= dd.buffer_steps) {
            FlushDeviceBufferForType(type);
            dd.start_step = step;
        }

        // Select field based on type
        ParGridFunction* field = nullptr;
        if (type == ReceiverType::Displacement) field = u_;
        else if (type == ReceiverType::Velocity) field = udot_;
        else if (type == ReceiverType::Acceleration || type == ReceiverType::Pressure) field = udot2_;
        if (!field) continue;

        const int num_local = dd.num_local;
        const int ncomp = dd.ncomp;
        const int buffer_steps = dd.buffer_steps;
        const int current_pos = dd.current_pos;

        const real_t* d_field = field->Read();
        const real_t* d_shape = dd.d_shape.Read();
        const int* d_vdofs = dd.d_vdofs.Read();
        const int* d_offsets = dd.d_offsets.Read();
        real_t* d_buffer = dd.d_buffer.ReadWrite();

        // Sign for pressure (negative)
        const real_t sign = (type == ReceiverType::Pressure) ? -1.0 : 1.0;

        // GPU kernel: interpolate and store in buffer
        MFEM_FORALL(idx, num_local * ncomp, {
            const int recv_idx = idx / ncomp;
            const int comp = idx % ncomp;

            const int offset = d_offsets[recv_idx];
            const int ndof = d_offsets[recv_idx + 1] - offset;

            real_t sum = 0.0;
            for (int k = 0; k < ndof; k++) {
                const int vdof_idx = offset * ncomp + comp * ndof + k;
                const int global_dof = d_vdofs[vdof_idx];
                sum += d_field[global_dof] * d_shape[offset + k];
            }

            const int buf_idx = recv_idx * buffer_steps * ncomp + current_pos * ncomp + comp;
            d_buffer[buf_idx] = sign * sum;
        });

        dd.current_pos++;
    }
}

void ReceiverArray::FlushDeviceBuffer() {
    for (auto& [type, dd] : device_data_) {
        if (dd.num_local > 0 && dd.current_pos > 0) {
            FlushDeviceBufferForType(type);
        }
    }
}

void ReceiverArray::FlushDeviceBufferForType(ReceiverType type) {
    auto it = device_data_.find(type);
    if (it == device_data_.end()) return;

    DeviceReceiverData& dd = it->second;
    if (dd.num_local == 0 || dd.current_pos == 0) return;

    // Read buffer from device to host
    const real_t* h_buffer = dd.d_buffer.HostRead();

    // Copy to ReceiverData
    auto& receivers = receivers_[type];
    int recv_idx = 0;

    for (auto& rec : receivers) {
        if (!rec.IsLocal()) continue;

        const int ncomp = dd.ncomp;
        const int buffer_steps = dd.buffer_steps;

        Vector sample(ncomp);
        for (int pos = 0; pos < dd.current_pos; pos++) {
            int step = dd.start_step + pos;
            for (int c = 0; c < ncomp; c++) {
                const int buf_idx = recv_idx * buffer_steps * ncomp + pos * ncomp + c;
                sample[c] = h_buffer[buf_idx];
            }
            rec.SetSample(step, sample);
        }

        recv_idx++;
    }

    // Reset buffer position
    dd.current_pos = 0;
}

// =============================================================================
// Device Initialization
// =============================================================================

void ReceiverArray::DeviceInit(int max_buffer_steps) {
    if (device_data_.empty()) {
        InitDeviceRecording(max_buffer_steps);
    }
}

}  // namespace SEM
