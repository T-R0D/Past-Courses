#ifndef _SIM_NODE_HPP_
#define _SIM_NODE_HPP_ 1

#include <vector>
#include <map>
#include <utility>
#include <functional>
#include <exception>
#include <cmath>

class SimNode {
 public:
  struct CoordinatePair {
    double x;
    double get_x() const
      {return x;};
    double y;
    double get_y() const
      {return y;};

    CoordinatePair() : x(0.0), y(0.0) {};
    CoordinatePair(const double _x, const double _y)
      {x = _x; y = _y;};
    bool operator<(const CoordinatePair& that) const
      {if (this->get_x() < that.get_x()) {return true;}
       else {return this->get_y() < that.get_y();}}
  };

  struct Target {
    double value;
    CoordinatePair location;

    Target() : value(0.0), location(0.0, 0.0) {};
    Target(const double _value, const CoordinatePair& _location)
      {value = _value; location = _location;};
  };

  struct TargetEstimate {
    double value;
    std::vector<double> consensus_weights;
  };

  struct EstimatePacket {
    int sender_id;
    CoordinatePair target_coordinates;
    double estimate;
  };

  struct RequestPacket {
    CoordinatePair target_location;
    std::vector<int> neighbor_ids;
  };

  static const double NO_READING;// = -1;

  SimNode();

  SimNode(
    const int p_id,
    const double p_sense_range,
    const double p_comm_range,
    const CoordinatePair& p_coordinates);

  ~SimNode();

  void TakeInitialReading(
    const double p_noisy_target_value,
    const CoordinatePair& p_target_coordinates);

  void PrepareForConsensusFinding(
    const std::function<std::vector<double>(
      const CoordinatePair&,
      const SimNode&,
      const std::map<int, SimNode>&)>& p_weight_design,
    const std::vector<SimNode>& p_nodes_in_network,
    const CoordinatePair& p_target_coordinates);

  void PerformConsensusIteration(
    const CoordinatePair& p_target_coordinates,
    const std::vector<EstimatePacket>& p_estimate_packets,
    const int p_consensus_iteration);

  RequestPacket RequestNeighborEstimates(
    const CoordinatePair& p_target_coordinates);

  RequestPacket RequestEstimateFromNearestNeighbor(
    const CoordinatePair& target_location);

  void RecieveNeighborEstimate(const EstimatePacket& neighbor_estimate);

  void AcceptNeighborEstimate(const EstimatePacket& neighbor_estimate);

  void CollectNeighborEstimates(
    const CoordinatePair& p_target_coordinates,
    const std::vector<EstimatePacket>& p_neighbor_estimates);

  EstimatePacket SendEstimatePacket(
    const CoordinatePair& p_target_location) const;


  int id() const;
  
  double sensing_range() const;

  double communication_range() const;
  CoordinatePair location() const;
  CoordinatePair coordinates() const;

  static double ComputeDistance(
    const CoordinatePair& p_source,
    const CoordinatePair& p_target);

  static double ComputeNoiseCovariance(
    const CoordinatePair& p_node_coordinates,
    const CoordinatePair& p_reckoning_point_coordinates,
    double p_weight_constant,
    double p_node_sensing_range);

 private:
  int id_;
  double sensing_range_;
  double communication_range_;
  CoordinatePair location_;
  std::map<CoordinatePair, TargetEstimate> target_estimates_;
  std::vector<int> neighbor_ids_;
  std::map<int, double> neighbor_weights_;
  double self_weight_;
};

#endif //_SIM_NODE_HPP_