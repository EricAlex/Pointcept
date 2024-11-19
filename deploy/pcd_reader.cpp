#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <limits>

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

struct PointIMOGTL
{
  PCL_ADD_POINT4D;
  float intensity;
  uint16_t laserid;
  double timeoffset;
  float yawangle;
  uint8_t mirrorid;
  uint16_t gt_label;
  PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
};

POINT_CLOUD_REGISTER_POINT_STRUCT (PointIMOGTL,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, intensity, intensity)
                                   (uint16_t, laserid, laserid)
                                   (double, timeoffset, timeoffset)
                                   (float, yawangle, yawangle)
                                   (uint8_t, mirrorid, mirrorid)
                                   (uint16_t, gt_label, gt_label)
)

struct PointIMOL
{
  PCL_ADD_POINT4D;
  float intensity;
  uint16_t laserid;
  double timeoffset;
  float yawangle;
  uint8_t mirrorid;
  uint16_t gtlabel;
  PCL_MAKE_ALIGNED_OPERATOR_NEW     // make sure our new allocators are aligned
};

POINT_CLOUD_REGISTER_POINT_STRUCT (PointIMOL,
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, intensity, intensity)
                                   (uint16_t, laserid, laserid)
                                   (double, timeoffset, timeoffset)
                                   (float, yawangle, yawangle)
                                   (uint8_t, mirrorid, mirrorid)
                                   (uint16_t, gtlabel, gtlabel)
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

Eigen::MatrixXd numpy_to_eigen(const py::array_t<double> &np_array) {
    Eigen::MatrixXd eigen_matrix(np_array.shape(0), np_array.shape(1));
    for (int i = 0; i < np_array.shape(0); ++i) {
        for (int j = 0; j < np_array.shape(1); ++j) {
            eigen_matrix(i, j) = np_array.at(i, j); 
        }
    }
    return eigen_matrix;
}

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

py::array_t<float> read_IMOL_pcd(const std::string &filename) {
    pcl::PointCloud<PointIMOL> cloud;

    if (pcl::io::loadPCDFile<PointIMOL>(filename, cloud) == -1) {
        throw std::runtime_error("Error loading PCD file!");
    }

    // Create a NumPy array with the appropriate shape and data type
    auto result = py::array_t<float>(std::vector<size_t>{cloud.size(), 5});

    // Access NumPy array's buffer for direct writing
    auto rinfo = result.request();
    float *result_ptr = (float *)rinfo.ptr;

    // Copy data from the PCL point cloud to the NumPy array
    for (size_t i = 0; i < cloud.size(); i++) {
        result_ptr[i * 5 + 0] = cloud.points[i].x;
        result_ptr[i * 5 + 1] = cloud.points[i].y;
        result_ptr[i * 5 + 2] = cloud.points[i].z;
        result_ptr[i * 5 + 3] = cloud.points[i].intensity;
        result_ptr[i * 5 + 4] = cloud.points[i].gtlabel;
    }

    return result;
}

py::array_t<float> read_IMOGTL_pcd(const std::string &filename) {
    pcl::PointCloud<PointIMOGTL> cloud;

    if (pcl::io::loadPCDFile<PointIMOGTL>(filename, cloud) == -1) {
        throw std::runtime_error("Error loading PCD file!");
    }

    // Create a NumPy array with the appropriate shape and data type
    auto result = py::array_t<float>(std::vector<size_t>{cloud.size(), 5});

    // Access NumPy array's buffer for direct writing
    auto rinfo = result.request();
    float *result_ptr = (float *)rinfo.ptr;

    // Copy data from the PCL point cloud to the NumPy array
    for (size_t i = 0; i < cloud.size(); i++) {
        result_ptr[i * 5 + 0] = cloud.points[i].x;
        result_ptr[i * 5 + 1] = cloud.points[i].y;
        result_ptr[i * 5 + 2] = cloud.points[i].z;
        result_ptr[i * 5 + 3] = cloud.points[i].intensity;
        result_ptr[i * 5 + 4] = cloud.points[i].gt_label;
    }

    return result;
}

py::array_t<float> read_IMOGTL_pcd_negative_selected_label(const std::string &filename,
                                                           const std::vector<size_t>& negative_selected_label,
                                                           const float valid_range) {
    pcl::PointCloud<PointIMOGTL> cloud;

    if (pcl::io::loadPCDFile<PointIMOGTL>(filename, cloud) == -1) {
        throw std::runtime_error("Error loading PCD file!");
    }

    double valid_range_sqaured = valid_range*valid_range;
    std::vector<size_t> filter_idx;
    for (size_t i = 0; i < cloud.size(); i++) {
        PointIMOGTL pt = cloud.points[i];
        bool is_in_negative_label(false), is_out_of_range(false);
        for(size_t j=0; j<negative_selected_label.size(); ++j){
            if(pt.gt_label==negative_selected_label[j]){
                is_in_negative_label = true;
                break;
            }
        }
        if((pt.x*pt.x)+(pt.y*pt.y) > valid_range_sqaured)
            is_out_of_range = true;
        if((!is_in_negative_label) && (!is_out_of_range))
            filter_idx.push_back(i);
    }

    // Create a NumPy array with the appropriate shape and data type
    auto result = py::array_t<float>(std::vector<size_t>{filter_idx.size(), 5});
    // Access NumPy array's buffer for direct writing
    auto rinfo = result.request();
    float *result_ptr = (float *)rinfo.ptr;
    // Copy data from the PCL point cloud to the NumPy array
    for (size_t i = 0; i < filter_idx.size(); i++) {
        result_ptr[i * 5 + 0] = cloud.points[filter_idx[i]].x;
        result_ptr[i * 5 + 1] = cloud.points[filter_idx[i]].y;
        result_ptr[i * 5 + 2] = cloud.points[filter_idx[i]].z;
        result_ptr[i * 5 + 3] = cloud.points[filter_idx[i]].intensity;
        result_ptr[i * 5 + 4] = cloud.points[filter_idx[i]].gt_label;
    }

    return result;
}

py::array_t<float> read_AL_pcd(const std::string &filename) {
    pcl::PointCloud<PointAL> cloud;

    if (pcl::io::loadPCDFile<PointAL>(filename, cloud) == -1) {
        throw std::runtime_error("Error loading PCD file!");
    }

    // Create a NumPy array with the appropriate shape and data type
    auto result = py::array_t<float>(std::vector<size_t>{cloud.size(), 5});

    // Access NumPy array's buffer for direct writing
    auto rinfo = result.request();
    float *result_ptr = (float *)rinfo.ptr;

    // Copy data from the PCL point cloud to the NumPy array
    for (size_t i = 0; i < cloud.size(); i++) {
        result_ptr[i * 5 + 0] = cloud.points[i].x;
        result_ptr[i * 5 + 1] = cloud.points[i].y;
        result_ptr[i * 5 + 2] = cloud.points[i].z;
        result_ptr[i * 5 + 3] = cloud.points[i].intensity;
        result_ptr[i * 5 + 4] = cloud.points[i].segLabel;
    }

    return result;
}

py::array_t<float> read_AL_pcd_with_excluded_area(const std::string &filename,
                                                py::array_t<double> &excluded_area,
                                                double ceiling_height) {
    pcl::PointCloud<PointAL> cloud;
    if (pcl::io::loadPCDFile<PointAL>(filename, cloud) == -1) {
        throw std::runtime_error("Error loading PCD file!");
    }

    Eigen::MatrixXd ex_area = numpy_to_eigen(excluded_area);
    std::vector<size_t> filter_idx;
    for (size_t i = 0; i < cloud.size(); i++) {
        PointAL pt = cloud.points[i];
        if(pt.x>ex_area(0, 0)&&pt.x<ex_area(0, 1)&&pt.y>ex_area(1, 0)&&pt.y<ex_area(1, 1))
            continue;
        if(pt.z>ceiling_height)
            continue;
        filter_idx.push_back(i);
    }

    // Create a NumPy array with the appropriate shape and data type
    auto result = py::array_t<float>(std::vector<size_t>{filter_idx.size(), 5});
    // Access NumPy array's buffer for direct writing
    auto rinfo = result.request();
    float *result_ptr = (float *)rinfo.ptr;
    // Copy data from the PCL point cloud to the NumPy array
    for (size_t i = 0; i < filter_idx.size(); i++) {
        result_ptr[i * 5 + 0] = cloud.points[filter_idx[i]].x;
        result_ptr[i * 5 + 1] = cloud.points[filter_idx[i]].y;
        result_ptr[i * 5 + 2] = cloud.points[filter_idx[i]].z;
        result_ptr[i * 5 + 3] = cloud.points[filter_idx[i]].intensity;
        result_ptr[i * 5 + 4] = cloud.points[filter_idx[i]].segLabel;
    }

    return result;
}

py::array_t<float> read_AL_pcd_selected_label(const std::string &filename,
                                            py::array_t<double> &excluded_area,
                                            double ceiling_height,
                                            double label_area_range,
                                            const std::vector<size_t>& selected_label) {
    pcl::PointCloud<PointAL> cloud;
    if (pcl::io::loadPCDFile<PointAL>(filename, cloud) == -1) {
        throw std::runtime_error("Error loading PCD file!");
    }

    Eigen::MatrixXd ex_area = numpy_to_eigen(excluded_area);
    std::vector<size_t> filter_idx;
    for (size_t i = 0; i < cloud.size(); i++) {
        PointAL pt = cloud.points[i];
        if(pt.z>ceiling_height)
            continue;
        if(pt.x<-label_area_range||pt.x>label_area_range||pt.y<-label_area_range||pt.y>label_area_range)
            continue;
        if(pt.x>ex_area(0, 0)&&pt.x<ex_area(0, 1)&&pt.y>ex_area(1, 0)&&pt.y<ex_area(1, 1))
            continue;
        bool is_in_label(false);
        for(size_t j=0; j<selected_label.size(); ++j){
            is_in_label = is_in_label||(pt.segLabel==selected_label[j]);
        }
        if(is_in_label)
            filter_idx.push_back(i);
    }

    // Create a NumPy array with the appropriate shape and data type
    auto result = py::array_t<float>(std::vector<size_t>{filter_idx.size(), 5});
    // Access NumPy array's buffer for direct writing
    auto rinfo = result.request();
    float *result_ptr = (float *)rinfo.ptr;
    // Copy data from the PCL point cloud to the NumPy array
    for (size_t i = 0; i < filter_idx.size(); i++) {
        result_ptr[i * 5 + 0] = cloud.points[filter_idx[i]].x;
        result_ptr[i * 5 + 1] = cloud.points[filter_idx[i]].y;
        result_ptr[i * 5 + 2] = cloud.points[filter_idx[i]].z;
        result_ptr[i * 5 + 3] = cloud.points[filter_idx[i]].intensity;
        result_ptr[i * 5 + 4] = cloud.points[filter_idx[i]].segLabel;
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

double p2lDistance(const Eigen::Vector2d& a, 
                              const Eigen::Vector2d& b, 
                              const Eigen::Vector2d& p){
    double delta_y = b(1) - a(1);
    double delta_x = b(0) - a(0);
    double numerator = fabs(delta_y*p(0) - delta_x*p(1) + b(0)*a(1) - b(1)*a(0));
    double denominator = sqrt(delta_y*delta_y + delta_x*delta_x);
    return numerator / denominator;
}

double Q_function(const double x){
  return exp(-(x*x)/2)/12 + exp(-2*(x*x)/3)/4;
}

double measurement_likelihood(const Eigen::Ref<const Eigen::Vector2d>& P, 
                                const Eigen::Ref<const Eigen::Vector2d>& eigenvalues,
                                const Eigen::Ref<const Eigen::Matrix2d>& eigenvectors,
                                const Eigen::Ref<const Eigen::Vector2d>& M,
                                const double sd_noise){
    double width(eigenvalues(0)), length(eigenvalues(1));
    double inner_base_weight(1/3.0);
    Eigen::Vector2d V_w = eigenvectors.col(0);
    Eigen::Vector2d V_l = eigenvectors.col(1);
    Eigen::Vector2d half_w_vec = 0.5*width*V_w;
    Eigen::Vector2d half_l_vec = 0.5*length*V_l;
    double d1 = p2lDistance(P + half_w_vec + half_l_vec, P + half_w_vec - half_l_vec, M);
    double d2 = p2lDistance(P - half_w_vec + half_l_vec, P - half_w_vec - half_l_vec, M);
    double Q1 = Q_function(d1/sd_noise);
    double Q2 = Q_function(d2/sd_noise);
    double f1(0), f2(0);
    f1 = inner_base_weight*fabs(Q1 - Q2);
    if((width>d1)&&(width>d2)){
        f1 = fabs(Q1 - Q2) + inner_base_weight;
    }
    d1 = p2lDistance(P + half_l_vec + half_w_vec, P + half_l_vec - half_w_vec, M);
    d2 = p2lDistance(P - half_l_vec + half_w_vec, P - half_l_vec - half_w_vec, M);
    Q1 = Q_function(d1/sd_noise);
    Q2 = Q_function(d2/sd_noise);
    f2 = inner_base_weight*fabs(Q1 - Q2);
    if((length>d1)&&(length>d2)){
        f2 = fabs(Q1 - Q2) + inner_base_weight;
    }
    return f1*f2/(length*width);
}

double objective_function(double yaw, const Eigen::MatrixXd& grid_points, double sd_noise) {
    const int num_points = grid_points.rows();

    Eigen::Matrix2d rotation_2d;
    rotation_2d << std::cos(yaw), -std::sin(yaw),
                   std::sin(yaw), std::cos(yaw);

    Eigen::MatrixXd rotated_point_cloud = grid_points * rotation_2d;
    Eigen::Vector2d min_coords = rotated_point_cloud.colwise().minCoeff();
    Eigen::Vector2d max_coords = rotated_point_cloud.colwise().maxCoeff();
    Eigen::Vector2d eigenvalues = max_coords - min_coords;
    Eigen::Vector2d center = (min_coords + max_coords) / 2.0;
    Eigen::Vector2d P =  center.transpose() * rotation_2d.transpose();
    double theta2 = yaw + M_PI/2;

    Eigen::Matrix2d eigenvectors;
    eigenvectors << std::cos(yaw), std::cos(theta2),
                   std::sin(yaw), std::sin(theta2);
    
    double likelihood_sum = 0.0;

    // Parallel likelihood calculation (use dynamic scheduling for load balancing)
    #pragma omp parallel for schedule(dynamic) reduction(+:likelihood_sum)
    for (int i = 0; i < num_points; ++i) {
        Eigen::Vector2d M = grid_points.row(i);
        likelihood_sum += measurement_likelihood(P, eigenvalues, eigenvectors, M, sd_noise);
    }

    return -likelihood_sum;
}

std::vector<double> fit_l_shape_3d(const Eigen::MatrixXd& grid_points, 
                                   const std::vector<double>& p_theta,
                                   double sd_noise) {

    std::vector<double> results(p_theta.size(), std::numeric_limits<double>::infinity());

    #pragma omp parallel for
    for (int i = 0; i < p_theta.size(); ++i) {
        results[i] = objective_function(p_theta[i], grid_points, sd_noise);
    }

    return results;
}


PYBIND11_MODULE(imo_pcd_reader, m) {
    m.doc() = "Module for reading PCD files with attributes using PyBind11";
    m.def("read_pcd", &read_pcd, "Reads a PCD file and returns a NumPy array");
    m.def("read_IMOL_pcd", &read_IMOL_pcd, "Reads half GT label PCD file and returns a NumPy array");
    m.def("read_IMOGTL_pcd", &read_IMOGTL_pcd, "Reads real GT label PCD file and returns a NumPy array");
    m.def("read_IMOGTL_pcd_negative_selected_label", &read_IMOGTL_pcd_negative_selected_label, "Reads real GT label PCD file and returns a NumPy array other than selected label");
    m.def("read_AL_pcd", &read_AL_pcd, "Reads a autolabel PCD file and returns a NumPy array");
    m.def("read_AL_pcd_with_excluded_area", &read_AL_pcd_with_excluded_area, "Reads a autolabel PCD file and returns a NumPy array outside the excluded area");
    m.def("read_AL_pcd_selected_label", &read_AL_pcd_selected_label, "Reads a autolabel PCD file and returns a NumPy array with selected label");
    m.def("save_pcd", &save_pcd, "Save a PCD file from NumPy arrays");
    m.def("save_imo_pcd", &save_imo_pcd, "Save to imotion PCD file from NumPy arrays");
    m.def("objective_function", &objective_function, "Calculate liklihood of point fitting to a bounding box");
    m.def("measurement_likelihood", &measurement_likelihood, "Calculate liklihood of point fitting to a bounding box");
    m.def("fit_l_shape_3d", &fit_l_shape_3d, "Calculate best L fitting of a bounding box");
}