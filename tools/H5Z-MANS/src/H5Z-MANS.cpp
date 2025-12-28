#include <vector>
#include <cstdlib> // for std::malloc, std::free
#include <cstring> // for std::memcpy
#include <new>     // for std::bad_alloc
#include <iostream>
#include <H5PLextern.h>


#include "mans_api.hpp"

// Define the Filter ID
#define H5Z_FILTER_MANS_ID 32001


// =========================================================
// Added: type check callback (can_apply)
// =========================================================
// Purpose: HDF5 calls this function before applying the filter to check whether
//          the datatype is supported.
// Return:  1 = allow, 0 = deny, -1 = error
static htri_t H5Z_can_apply_mans(hid_t dcpl_id, hid_t type_id, hid_t space_id)
{
    if (H5Tget_class(type_id) != H5T_INTEGER) {
        std::cerr << "[H5Z-MANS Warning] Datatype is not INTEGER.\n";
        return 0;
    }
    H5T_sign_t sign = H5Tget_sign(type_id);
    if (sign != H5T_SGN_NONE) {
        std::cerr << "[H5Z-MANS Warning] Datatype must be Unsigned (UINT).\n";
        return 0;
    }
    size_t size = H5Tget_size(type_id);
    if (size != 2 && size != 4) {
        std::cerr << "[H5Z-MANS Warning] Only 2-byte (U16) or 4-byte (U32) supported. Current: " << size << "\n";
        return 0;
    }
    // Passed all checks, allow this filter
    return 1;
}

// =========================================================
// Filter callback function: H5Z_filter_mans
// =========================================================
// Parameter notes:
// flags:     H5Z_FLAG_REVERSE means decompress; otherwise compress
// cd_nelmts: length of cd_values array
// cd_values: configuration parameter array (corresponds to MansParams)
// nbytes:    input data size in bytes
// buf_size:  total capacity of the input buffer (may be > nbytes)
// buf:       pointer to the data pointer (if we need to grow it, we modify *buf)
// Return:    on success, number of output bytes; on failure, 0
static size_t H5Z_filter_mans(unsigned int flags, size_t cd_nelmts,
                             const unsigned int cd_values[], size_t nbytes,
                             size_t *buf_size, void **buf)
{
    size_t required_params = sizeof(mans::MansParams) / sizeof(unsigned int);

    if (cd_nelmts < required_params) {
        // Not enough parameters to restore configuration; treat as a severe error
        return 0;
    }
    const mans::MansParams* params = reinterpret_cast<const mans::MansParams*>(cd_values);
    std::vector<uint8_t> out_buf;
    try {
        if (flags & H5Z_FLAG_REVERSE) {
            mans::decompress(*buf, nbytes, *params, out_buf);
        } else {
            size_t elem_size = (params->dtype == mans::DataType::U16) ? 2 : 4;
            if (nbytes % elem_size != 0) {
                // Data size is not a multiple of element size; likely invalid
                return 0;
            }
            size_t length = nbytes / elem_size;
            mans::compress(*buf, length, *params, out_buf);
        }

        // ==========================================
        // HDF5 memory management rules
        // ==========================================
        // Rules:
        // 1. If output size (out_buf.size()) is larger than current buffer capacity (*buf_size),
        //    free the old memory, allocate new memory, and update *buf_size.
        // 2. If output size is smaller than current capacity, we can reuse the old buffer
        //    (shrinking is optional).
        // 3. Must use standard malloc/free (unless HDF5 was built with a custom allocator).
        if (out_buf.size() > *buf_size) {
            if (*buf) std::free(*buf); 
            *buf = std::malloc(out_buf.size()); 
            if (!*buf) {
                return 0; // Out Of Memory!
            }
            *buf_size = out_buf.size(); 
        }
        if (!out_buf.empty()) {
            std::memcpy(*buf, out_buf.data(), out_buf.size());
        }

        
        return out_buf.size();

    } catch (const std::exception& e) {
        std::cerr << "[H5Z-MANS Error]: " << e.what() << "\n";
        return 0;
    } catch (...) {
        std::cerr << "[H5Z-MANS Error]: Unknown exception occurred.\n";
        return 0;
    }
}

// =========================================================
// HDF5 plugin registration structure
// =========================================================
const H5Z_class2_t H5Z_MANS_CLASS[1] = {{
    H5Z_CLASS_T_VERS,       
    H5Z_FILTER_MANS_ID,     // Filter ID (32001)
    1,                      
    1,                      
    "H5Z-MANS",             
    H5Z_can_apply_mans,     
    NULL,                   
    H5Z_filter_mans,        
}};

H5PL_type_t H5PLget_plugin_type(void) {
    return H5PL_TYPE_FILTER;
}
const void *H5PLget_plugin_info(void) {
    return H5Z_MANS_CLASS;
}