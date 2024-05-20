#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace py = pybind11;

struct PointIMO
{
  PCL_ADD_POINT4D;
  float intensity;
  uint16_t laserid;
  double timeoffset;
  float yawangle;
  uint8_t mirrorid;
  PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
};

POINT_CLOUD_REGISTER_POINT_STRUCT (PointIMO,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, intensity, intensity)
                                   (uint16_t, laserid, laserid)
                                   (double, timeoffset, timeoffset)
                                   (float, yawangle, yawangle)
                                   (uint8_t, mirrorid, mirrorid)
)

struct PointAL
{
  PCL_ADD_POINT4D;
  float intensity;
  uint16_t segLabel;
  PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
};

POINT_CLOUD_REGISTER_POINT_STRUCT (PointAL,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, intensity, intensity)
                                   (uint16_t, segLabel, segLabel)
)

// Function to read a PCD file and convert it to a NumPy array
py::array_t<float> read_pcd(const std::string &filename) {
    pcl::PointCloud<PointIMO> cloud;

    if (pcl::io::loadPCDFile<PointIMO>(filename, cloud) == -1) {
        throw std::runtime_error("Error loading PCD file!");
    }

    // Create a NumPy array with the appropriate shape and data type
    auto result = py::array_t<float>(std::vector<size_t>{cloud.size(), 4});

    // Access NumPy array's buffer for direct writing
    auto rinfo = result.request();
    float *result_ptr = (float *)rinfo.ptr;

    // Copy data from the PCL point cloud to the NumPy array
    for (size_t i = 0; i < cloud.size(); i++) {
        result_ptr[i * 4 + 0] = cloud.points[i].x;
        result_ptr[i * 4 + 1] = cloud.points[i].y;
        result_ptr[i * 4 + 2] = cloud.points[i].z;
        result_ptr[i * 4 + 3] = cloud.points[i].intensity;
    }

    return result;
}

void save_pcd(py::array_t<float> &coord_intensities,
              py::array_t<float> &segLabels,
              const std::string &filename) {
    // Ensure NumPy arrays have the right shapes and types
    if (coord_intensities.ndim() != 2 || coord_intensities.shape(1) != 4)
        throw std::runtime_error("Coord-intensity array must have shape (N, 4)");
    if (segLabels.ndim() != 1 || segLabels.shape(0) != coord_intensities.shape(0)){
        std::cout<<"Label dim: "<<segLabels.shape(0)<<", coords dim: "<<coord_intensities.shape(0)<<std::endl;
        throw std::runtime_error("SegLabels array must have shape (N,)");
    }

    // Create PCL point cloud with intensity
    pcl::PointCloud<PointAL> cloud; 
    cloud.width = coord_intensities.shape(0);
    cloud.height = 1; // Unorganized
    cloud.is_dense = false;
    cloud.points.resize(cloud.width * cloud.height);

    // Fill the point cloud data
    auto coord_i_ptr = coord_intensities.unchecked();
    auto segLabels_data_ptr = segLabels.unchecked();
    for (size_t i = 0; i < cloud.points.size(); ++i) {
        cloud.points[i].x = coord_i_ptr(i, 0);
        cloud.points[i].y = coord_i_ptr(i, 1);
        cloud.points[i].z = coord_i_ptr(i, 2);
        cloud.points[i].intensity = coord_i_ptr(i, 3);
        cloud.points[i].segLabel = segLabels_data_ptr(i);
    }

    // Save the PCD file
    pcl::io::savePCDFileBinary(filename, cloud);
}

void save_imo_pcd(py::array_t<float> &coord_intensities,
                  const std::string &filename) {
    // Ensure NumPy arrays have the right shapes and types
    if (coord_intensities.ndim() != 2 || coord_intensities.shape(1) != 4)
        throw std::runtime_error("Coord-intensity array must have shape (N, 4)");

    // Create PCL point cloud with intensity
    pcl::PointCloud<PointIMO> cloud; 
    cloud.width = coord_intensities.shape(0);
    cloud.height = 1; // Unorganized
    cloud.is_dense = false;
    cloud.points.resize(cloud.width * cloud.height);

    // Fill the point cloud data
    auto coord_i_ptr = coord_intensities.unchecked();
    for (size_t i = 0; i < cloud.points.size(); ++i) {
        cloud.points[i].x = coord_i_ptr(i, 0);
        cloud.points[i].y = coord_i_ptr(i, 1);
        cloud.points[i].z = coord_i_ptr(i, 2);
        cloud.points[i].intensity = coord_i_ptr(i, 3);
        cloud.points[i].laserid = 0;
        cloud.points[i].timeoffset = 0;
        cloud.points[i].yawangle = 0;
        cloud.points[i].mirrorid = 0;
    }

    // Save the PCD file
    pcl::io::savePCDFileBinary(filename, cloud);
}

PYBIND11_MODULE(imo_pcd_reader, m) {
    m.doc() = "Module for reading PCD files with attributes using PyBind11";
    m.def("read_pcd", &read_pcd, "Reads a PCD file and returns a NumPy array");
    m.def("save_pcd", &save_pcd, "Save a PCD file from NumPy arrays");
    m.def("save_imo_pcd", &save_imo_pcd, "Save to imotion PCD file from NumPy arrays");
}