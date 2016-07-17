#ifndef _SIM_NODE_
#define _SIM_NODE_ 1

#include "sim_node.hpp"

#include <vector>
#include <map>
#include <utility>
#include <functional>
#include <exception>
#include <cmath>
#include <iostream>

const double SimNode::NO_READING = 0;

SimNode::SimNode() {/*TODO*/}

SimNode::SimNode(
  const int p_id,
  const double p_sense_range,
  const double p_comm_range,
  const CoordinatePair& p_coordinates) :
  id_(p_id),
  sensing_range_(p_sense_range),
  communication_range_(p_comm_range),
  location_(p_coordinates),
  target_estimates_(),
  neighbor_ids_() {}

SimNode::~SimNode() {}

void
SimNode::TakeInitialReading(
  const double p_noisy_target_value,
  const CoordinatePair& p_target_coordinates) {

  double distance = ComputeDistance(location_, p_target_coordinates);
  if (distance <= sensing_range_) {
    target_estimates_[p_target_coordinates].value = p_noisy_target_value;
  } else {
    target_estimates_[p_target_coordinates].value = NO_READING;
  }
}

void
SimNode::PrepareForConsensusFinding(
  const std::function<double(SimNode, SimNode)>& p_weight_design,
  const std::vector<SimNode>& p_nodes_in_network,
  const CoordinatePair& p_target_coordinates) {

  neighbor_ids_.clear();
  for (SimNode node : p_nodes_in_network) {
    if (node.id() != id_) {
      double separation_distance = ComputeDistance(node.location(), location_);

      if (separation_distance <= node.communication_range() &&
          separation_distance <= communication_range_) {
        double d = ComputeDistance(node.location(), p_target_coordinates);

        if (d <= node.sensing_range()) {
          neighbor_ids_.push_back(node.id());
        }
      }
    }
  }

for (int id : neighbor_ids_) {
std::cout << id << ' ';
}
std::cout << std::endl;

  target_estimates_[p_target_coordinates].value = ;
  estimate.consensus_weights =

  neighbor_weights_.clear();
  for (SimNode neighbor : p_nodes_in_network) {
    for (int id : neighbor_ids_) {
      if (neighbor.id() == id) {
        neighbor_weights_[neighbor.id()] = p_weight_design(*this, neighbor);
      }
    }
  }
  self_weight_ = p_weight_design(*this, *this);
}

void
SimNode::PerformConsensusIteration(
    const CoordinatePair& p_target_coordinates,
    const std::vector<EstimatePacket>& p_estimate_packets,
    const int p_consensus_iteration) {

  // for the simulation, the iterations are handled by the node network
  // ...IRL though, work would need to be done here
}

SimNode::RequestPacket
SimNode::RequestNeighborEstimates(
  const CoordinatePair& p_target_coordinates) {

  RequestPacket request;
  request.target_location = p_target_coordinates;

  for (int neighbor_id : neighbor_ids_) {
    request.neighbor_ids.push_back(neighbor_id);
  }

  return request;
}

SimNode::RequestPacket
SimNode::RequestEstimateFromNearestNeighbor(
  const CoordinatePair& target_location) {

  if (neighbor_ids_.empty()) {
    throw std::exception();
  }

  RequestPacket request;
  request.target_location = target_location;
  request.neighbor_ids.push_back(neighbor_ids_.front());// TODO: make this real

  return request; 
}

void
SimNode::RecieveNeighborEstimate(const EstimatePacket& neighbor_estimate) {
  target_estimates_[neighbor_estimate.target_coordinates] =
    neighbor_estimate.estimate;                                ///////////////////////////////TODO
}

void
SimNode::AcceptNeighborEstimate(const EstimatePacket& neighbor_estimate) {
  target_estimates_[neighbor_estimate.target_coordinates] =
    neighbor_estimate.estimate;
}


void
SimNode::CollectNeighborEstimates(
  const CoordinatePair& p_target_coordinates,
  const std::vector<EstimatePacket>& p_neighbor_estimates) {


                        ////////////////////////////////////// TODO

}

SimNode::EstimatePacket
SimNode::SendEstimatePacket(
  const CoordinatePair& p_target_location) const {

  EstimatePacket packet;
  packet.sender_id = id_;
  packet.target_coordinates = p_target_location;
  packet.estimate = target_estimates_.at(p_target_location);

  return packet;
}

int
SimNode::id() const {
  return id_;
}

double
SimNode::sensing_range() const {
  return sensing_range_;
}

double
SimNode::communication_range() const {
  return communication_range_;
}

SimNode::CoordinatePair
SimNode::location() const {
  return location_;
}

SimNode::CoordinatePair
SimNode::coordinates() const {
  return location_;
}

double
SimNode::ComputeDistance(
  const CoordinatePair& p_source,
  const CoordinatePair& p_target) {

  // Euclidean Distance = sqrt((x_0 - x_1)^2 + (y_0 - y_1)^2)

  double x_diff = p_source.x - p_target.x;
  double y_diff = p_source.y - p_target.y;

  return sqrt((x_diff * x_diff) + (y_diff * y_diff));
}

double
SimNode::ComputeNoiseCovariance(
  const CoordinatePair& p_node_coordinates,
  const CoordinatePair& p_reckoning_point_coordinates,
  double p_weight_constant,
  double p_node_sensing_range) {

  double distance = ComputeDistance(
    p_node_coordinates,
    p_reckoning_point_coordinates
  );

  double numerator = (distance * distance) + p_weight_constant;

  return numerator / (p_node_sensing_range * p_node_sensing_range);
}

#endif //_SIM_NODE_