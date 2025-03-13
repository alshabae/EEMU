#include <iostream>
#include <eigen3/Eigen/Dense>
#include <random>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include <pybind11/pybind11.h>

using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorMatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorVectorXi = Eigen::Matrix<int, 1, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorVectorSliceXi = Eigen::IndexedView<Eigen::RowVectorXi, Eigen::internal::SingleRange, Eigen::ArithmeticSequence<Eigen::Index, Eigen::Index, Eigen::Index>>;

class Sampler {
    private:
        std::mt19937 gen;
        std::uniform_real_distribution<float> dist;
        std::vector<std::vector<float>> probs;
        int max_num_neighbors;


    public:
        RowMajorVectorXi indptr;
        RowMajorVectorXi degrees;
        RowMajorVectorXi data;

        Sampler(){}
        Sampler(const Eigen::Ref<const RowMajorVectorXi>& indptr, 
                const Eigen::Ref<const RowMajorVectorXi>& degrees, 
                const Eigen::Ref<const RowMajorVectorXi>& data, 
                int seed){
            this->gen = std::mt19937(seed);
            this->dist = std::uniform_real_distribution<float>(0, 1);
            this->indptr = indptr;
            this->degrees = degrees;
            this->data = data;
        }

        void SetNeighborhoodProbabilities(const std::vector<std::vector<float>>& probs){
            this->probs = probs;
            auto max_neighborhood_vec_iter = std::max_element(this->probs.begin(), 
                                                         this->probs.end(), 
                                                         [](std::vector<float>& lhs, std::vector<float>& rhs){ return lhs.size() < rhs.size(); });
            this->max_num_neighbors = max_neighborhood_vec_iter[0].size();
        }

        int Length(){
            return data.size();
        }
        
        RowMajorVectorSliceXi operator[](int index){
            int start_ind = this->indptr(index);
            int end_ind = start_ind + this->degrees(index);
            return this->data(Eigen::seq(start_ind, end_ind - 1, 1));
        }

        RowMajorMatrixXi SampleNaive(const Eigen::Ref<const RowMajorVectorXi>& nodes, int num_samples){ 
            int num_nodes = nodes.cols();
            RowMajorMatrixXi samples(num_nodes, num_samples);

            #pragma omp parallel for num_threads(16)
            for(int n = 0; n < num_nodes; n++){
                int node = nodes(n);
                auto probs_for_node = this->probs[node];
                auto neighbors_for_node = (*this)[node];
                int num_neighbors = probs_for_node.size();

                // num_samples must be <= num_neighbors
                if(num_samples > num_neighbors) 
                    throw std::invalid_argument("num_samples must be <= num_neighbors for all nodes in the batch");

                for(int i = 0; i < num_samples; i++){
                    std::discrete_distribution<int> d(probs_for_node.begin(), probs_for_node.end());
                    int sampled_id = d(this->gen);
                    probs_for_node[sampled_id] = 0;
                    samples(n, i) = neighbors_for_node(sampled_id);
                }
            }
            return samples;
        }

        RowMajorMatrixXi SampleOptimized(const Eigen::Ref<const RowMajorVectorXi>& nodes, int num_samples){
            int num_nodes = nodes.cols();
            RowMajorMatrixXi samples(num_nodes, num_samples);

            // The same random uniform samples can be used for each node's sampling process
            std::vector<float> uniform_samples(this->max_num_neighbors);
            std::generate(uniform_samples.begin(), uniform_samples.end(), [&](){ return this->dist(this->gen); });

            std::vector<float> res(this->max_num_neighbors);
            std::vector<int> indices(this->max_num_neighbors);
            for(int n = 0;  n < num_nodes; n++){
                int node  = nodes(n);
                auto weights = this->probs[node]; // (num_neighbors,)
                auto neighbors_for_node = (*this)[node];
                int num_neighbors = weights.size();

                // num_samples must be <= num_neighbors
                if(num_samples > num_neighbors)
                    throw std::invalid_argument("num_samples must be <= num_neighbors for all nodes in the batch");

                // Compute 1 / wi
                std::transform(weights.begin(), weights.end(), res.begin(), [](float wi){ return 1 / wi; });
                // Compute ui ^ (1 / wi)
                std::transform(uniform_samples.begin(), 
                               uniform_samples.begin() + num_neighbors, 
                               res.begin(), 
                               res.begin(), 
                               [](float ui, float inv_wi){ return std::pow(ui, inv_wi); });
                
                // Sort keys
                std::iota(indices.begin(), indices.begin() + num_neighbors, 0);
                std::stable_sort(indices.begin(), indices.begin() + num_neighbors, [&](int i, int j){ return res[i] > res[j]; }); 

                // Write top num_samples samples to row  in samples matrix
                std::transform(indices.begin(), indices.begin() + num_samples, samples.data() + n * num_samples, [&](int i){ return neighbors_for_node(i); });          
            }

            return samples;
        }

        RowMajorMatrixXi SampleOptimizedOMP(const Eigen::Ref<const RowMajorVectorXi>& nodes, int num_samples){
            int num_nodes = nodes.cols();
            RowMajorMatrixXi samples(num_nodes, num_samples);

            // The same random uniform samples can be used for each node's sampling process
            std::vector<float> uniform_samples(this->max_num_neighbors);
            std::generate(uniform_samples.begin(), uniform_samples.end(), [&](){ return this->dist(this->gen); });

            pybind11::gil_scoped_release relesase;
            #pragma omp parallel num_threads(16)
            {
                std::vector<float> res(this->max_num_neighbors);
                std::vector<int>  indices(this->max_num_neighbors);
                #pragma omp for
                for(int n = 0;  n < num_nodes; n++){
                    int node  = nodes(0, n);
                    auto weights = this->probs[node]; // (num_neighbors,)
                    auto neighbors_for_node = (*this)[node];
                    int num_neighbors = weights.size();

                    // num_samples must be <= num_neighbors
                    if(num_samples > num_neighbors)
                        throw std::invalid_argument("num_samples must be <= num_neighbors for all nodes in the batch");

                    // Compute 1 / wi
                    std::transform(weights.begin(), weights.end(), res.begin(), [](float wi){ return 1 / wi; });
                    // Compute ui ^ (1 / wi)
                    std::transform(uniform_samples.begin(), 
                                uniform_samples.begin() + num_neighbors, 
                                res.begin(), 
                                res.begin(), 
                                [](float ui, float inv_wi){ return std::pow(ui, inv_wi); });
                    
                    // Sort keys
                    std::iota(indices.begin(), indices.begin() + num_neighbors, 0);
                    std::stable_sort(indices.begin(), indices.begin() + num_neighbors, [&](int i, int j){ return res[i] > res[j]; }); 

                    // Write top num_samples samples to row  in samples matrix
                    std::transform(indices.begin(), indices.begin() + num_samples, samples.data() + n * num_samples, [&](int i){ return neighbors_for_node(i); });                            
                }
            }

            pybind11::gil_scoped_acquire acquire;
            return samples;
        }        
};