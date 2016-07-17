#ifndef _SIM_NODE_NETWORK_HPP_
#define _SIM_NODE_NETWORK_HPP_ 1

#include <functional>
#include <vector>
#include <list>
#include <random>

#include <Eigen/Dense>

#include "sim_node.hpp"

class SimNodeNetwork {
 public:
  typedef std::vector<double> IterationEstimate;

  struct EstimateLog {
    SimNode::Target target;
    std::vector<IterationEstimate> estimates;

    EstimateLog() : target(0.0, {0.0, 0.0}), estimates() {};
  };

  static const double NO_READING;

  static std::vector<double> MaxDegreeConsensus(
    const SimNode::CoordinatePair& target_location,
    const SimNode& source_node,
    const std::vector<SimNode>& nodes_in_network);

  // static const std::function<double(int, int)> metropolis_consensus =
  //   [] (int p_source_vertex, int p_SimNode::Target_vertex) -> double {
  //       return 0.0;
  //   }

  SimNodeNetwork(
    const std::function<double(SimNode, SimNode)>& p_consensus_method,
    const double p_error_threshold);

  ~SimNodeNetwork();

  SimNodeNetwork& AddNode(
    const double p_sensing_range,
    const double p_communication_range,
    const SimNode::CoordinatePair& p_coordinates);

  std::vector<EstimateLog> BuildMapOfField(
    const std::vector<SimNode::Target>& targets);

  void PerformSensorSampling(
    const SimNode::Target& target);

  void PrepareNodesForConsensusFiltering(
    const SimNode::Target& target);

  EstimateLog PerformConsensusFiltering(
    const SimNode::Target& target);

  std::vector<SimNode> FindSubSetOfNodes(
    const std::map<int, SimNode>& sensor_nodes,
    const SimNode::CoordinatePair& target_location,
    const std::function<bool(
      const SimNode&,
      const SimNode::CoordinatePair&
    )>& CriteriaIsMet);

  std::vector<double> CollectEstimatesFromNodes(
    const SimNode::CoordinatePair& target_location);

  double ComputeAverageReading(
    const std::vector<SimNode> estimating_nodes,
    const SimNode::CoordinatePair& target_location);

  bool EstimatesHaveSufficientlyConverged(
    const std::vector<double>& estimates,
    const double average_reading) const;

  bool EstimatesHaveNotSufficientlyConverged(
    const std::vector<double>& estimates,
    const double average_reading) const;


 private:
  int next_node_id_;
  double error_threshold_;
  std::map<int, SimNode> sensor_nodes_;
  SimNode::CoordinatePair average_node_location_;
  std::vector<SimNode::CoordinatePair> target_locations_;
  std::map<SimNode::CoordinatePair, double> average_readings_; // aka convergence values
  std::function<double(SimNode, SimNode)> weight_strategy_;
};

#endif //_SIM_NODE_NETWORK_HPP_