#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "sampler.h"

PYBIND11_MODULE(cpp_ops, m){
    m.doc() = "Custom C++ sampler";

    pybind11::class_<Sampler>(m, "Sampler")
    .def(pybind11::init<const Eigen::Ref<const RowMajorVectorXi>&, const Eigen::Ref<const RowMajorVectorXi>&, const Eigen::Ref<const RowMajorVectorXi>&, int>())
    .def("__len__", &Sampler::Length)
    .def("__getitem__", &Sampler::operator[])
    .def("SetNeighborhoodProbabilities", &Sampler::SetNeighborhoodProbabilities)
    .def("SampleNaive",  &Sampler::SampleNaive, pybind11::return_value_policy::take_ownership, pybind11::call_guard<pybind11::gil_scoped_release>())
    .def("SampleOptimized", &Sampler::SampleOptimized, pybind11::return_value_policy::take_ownership)
    .def("SampleOptimizedOMP", &Sampler::SampleOptimizedOMP, pybind11::return_value_policy::take_ownership);
}


