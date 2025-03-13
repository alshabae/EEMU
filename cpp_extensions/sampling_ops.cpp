#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>

using RowMajorMatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorVectorXi = Eigen::Matrix<int, 1, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorVectorSliceXi = Eigen::IndexedView<Eigen::RowVectorXi, Eigen::internal::SingleRange, Eigen::ArithmeticSequence<Eigen::Index, Eigen::Index, Eigen::Index>>;

int main(){
    // RowMajorMatrixXi m(10, 15);
    // RowMajorMatrixXi n(10, 15);
    // std::cout <<  m  <<  std::endl;

    // RowMajorVectorXi block =  m(0, Eigen::all);
    // block(0, 1) = 10;

    // std::cout <<  block  <<  std::endl;

    // std::cout  <<  m  << std::endl;

    // std::vector<std::vector<int>> a;
    // for (int i = 0; i < 10; i++){
    //     std::vector<int> b(i + 7);
    //     a.push_back(b);
    // }

    // auto max_len_vec = std::max_element(a.begin(), a.end(), [](std::vector<int>& lhs, std::vector<int>& rhs) { return lhs.size() < rhs.size(); });
    // std::cout << max_len_vec[0].size() << std::endl;

    RowMajorVectorXi a(10);
    std::iota(a.data(), a.data() + a.size(), 0);
    RowMajorVectorSliceXi b = a(Eigen::seq(0, a.size() - 1, 2));
    b(0) = 99;
    std::cout << b << std::endl;
    std::cout << a << std::endl;

    // std::vector<int> a;
    // for(int i = 0; i < 15; i++) a.push_back(i);

    // auto start = std::chrono::high_resolution_clock::now();
    // for(int c = 0; c < 50000; c++)
    //     for(int i = 0; i < 15; i++)
    //         m(0, i) = a[i];
    // auto stop  = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop -  start);
    // std::cout << "Looping duration count = " << duration.count() << std::endl;

    // start = std::chrono::high_resolution_clock::now();
    // for(int c = 0; c < 50000; c++)
    //     std::copy(a.begin(), a.end(), n.data());
    // stop  = std::chrono::high_resolution_clock::now();
    // duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop -  start);
    // std::cout << "std::copy duration count = " << duration.count() << std::endl;

    // std::cout << m << std::endl;
    // std::cout << '\n';
    // std::cout << n << std::endl;


    // std::vector<int> indices(6);
    // std::vector<int> vals = {8, 2, 11, 10, 3, 17, 6, 12, 15, 0};
    // std::iota(indices.begin(), indices.end(), 0);
    // std::stable_sort(indices.begin(), indices.end(), [&](int i, int j){ return vals[i] > vals[j]; }); 
    // for(int &i : indices) std::cout << vals[i] << " ";
    // std::cout << '\n';

    // for(int &i : indices) std::cout << i;
    // std::cout << '\n';
}
