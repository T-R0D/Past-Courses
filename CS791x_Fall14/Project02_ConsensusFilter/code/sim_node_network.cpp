#ifndef _SIM_NODE_NETWORK_CPP_
#define _SIM_NODE_NETWORK_CPP_ 1

#include "sim_node_network.hpp"

#include <vector>
#include <list>
#include <exception>
#include <iostream>

const unsigned int SEED = 1;

const double SimNodeNetwork::NO_READING = SimNode::NO_READING;

std::vector<double>
SimNodeNetwork::MaxDegreeConsensus(
    const SimNode::CoordinatePair& target_location,
    const SimNode& source_node,
    const std::map<int, SimNode>& nodes_in_network) {

  std::vector<SimNode::EstimatePacket> estimates;

  for (SimNode node : nodes_in_network) {
    double comm_distance = SimNode::ComputeDistance(
      source_node.location(),
      node.location()
    );

    if (comm_distance <= source_node.communication_range() &&
        comm_distance <= node.communication_range()) {
      Simnode::EstimatePacket packet = node.SendEstimatePacket(target_location)
      estimates.push_back(packet);
    }
  }

  std::vector<double> weights(nodes_in_network.size());
  double neighbor_weight_sum = 0.0;
  for (double weight : weights) {
    weight = 0.0;
  }
  
  for (EstimatePacket packet : estimates) {
    if (packet.sender_id != source_node.id()) {
      if (packet.estimate != NO_READING) {
        weights[packet.sender_id] =
          1.0 / (double) nodes_in_network.size();

        neighbor_weight_sum += weights[packet.sender_id];
      } else {
        weights[packet.sender_id] = 0.0;
      }
    }
  }

  weights[source_node.id()] = 1.0 - neighbor_weight_sum;

  return weights;
}

SimNodeNetwork::SimNodeNetwork(
  const std::function<double(SimNode, SimNode)>& p_consensus_method,
  const double p_error_threshold) :
  next_node_id_(0),
  error_threshold_(p_error_threshold),
  sensor_nodes_(),
  average_node_location_(0, 0),
  target_locations_(),
  average_readings_(),
  weight_strategy_(p_consensus_method) {}

SimNodeNetwork::~SimNodeNetwork() {}

SimNodeNetwork&
SimNodeNetwork::AddNode(
  const double p_sensing_range,
  const double p_communication_range,
  const SimNode::CoordinatePair& p_coordinates) {

  SimNode new_sensor_node(
    next_node_id_,
    p_sensing_range,
    p_communication_range,
    p_coordinates
  );
  next_node_id_++;

  average_node_location_.x *= (
    (double) sensor_nodes_.size() / (double) (sensor_nodes_.size() + 1)
  );
  average_node_location_.y *= (
    (double) sensor_nodes_.size() / (double) (sensor_nodes_.size() + 1)
  );
  average_node_location_.x += (
    p_coordinates.x / (double) (sensor_nodes_.size() + 1)
  );
  average_node_location_.y += (
    p_coordinates.y / (double) (sensor_nodes_.size() + 1)
  );

  sensor_nodes_[new_sensor_node.id()] = new_sensor_node;
}

std::vector<SimNodeNetwork::EstimateLog>
SimNodeNetwork::BuildMapOfField(
  const std::vector<SimNode::Target>& targets) {

std::cout << "average node location: " << average_node_location_.x << ' ' << average_node_location_.y << std::endl;

  std::vector<EstimateLog> results;

  for (SimNode::Target target : targets) {
    EstimateLog result;

    PerformSensorSampling(target); // SimNode::Target sampling

    PrepareNodesForConsensusFiltering(target); // weight estimation

    result = PerformConsensusFiltering(target); // consensus finding

    results.push_back(result);
  }

  return results;
}

void
SimNodeNetwork::PerformSensorSampling(
  const SimNode::Target& target) {

  std::default_random_engine generator(SEED);
  
  for (auto entry : sensor_nodes_) {
    SimNode node = entry.second;

    std::normal_distribution<double> noise_distribution(
      target.value,
      SimNode::ComputeNoiseCovariance(
        node.location(),
        average_node_location_,
        0.001,
        node.sensing_range()
      )
    );

    double noisy_reading = noise_distribution(generator);

    node.TakeInitialReading(
      noisy_reading,
      target.location
    );

std::cout << noisy_reading << std::endl;
  }
}

void
SimNodeNetwork::PrepareNodesForConsensusFiltering(
  const SimNode::Target& target) {

  std::vector<SimNode> node_vector(sensor_nodes_.size());
  for (auto entry : sensor_nodes_) {
    node_vector.push_back(sensor_nodes_[entry.first]);
  }

  for (SimNode node : node_vector) {
    node.PrepareForConsensusFinding(
      weight_strategy_,
      node_vector,
      target.location
    );
  }
}

SimNodeNetwork::EstimateLog
SimNodeNetwork::PerformConsensusFiltering(
  const SimNode::Target& target) {

  SimNode::CoordinatePair target_location = target.location;

  std::vector<SimNode> estimating_nodes = FindSubSetOfNodes(
    sensor_nodes_,
    target_location,
    [] (const SimNode& node, const SimNode::CoordinatePair& target_location) -> bool {
      return (node.SendEstimatePacket(target_location)).estimate != NO_READING;
    }
  );

  std::vector<SimNode> passive_nodes = FindSubSetOfNodes(
    sensor_nodes_,
    target_location,
    [] (const SimNode& node, const SimNode::CoordinatePair& target_location) -> bool {
      return (node.SendEstimatePacket(target_location)).estimate == NO_READING;
    }
  );

  EstimateLog estimate_log;
  estimate_log.target = target;

  std::vector<double> estimates = CollectEstimatesFromNodes(target_location);
  estimate_log.estimates.push_back(estimates);

  double average_reading = ComputeAverageReading(
    estimating_nodes,
    target_location
  );

  while (EstimatesHaveNotSufficientlyConverged(estimates, average_reading)) {
    for (SimNode node : estimating_nodes) {
      SimNode::RequestPacket request = node.RequestNeighborEstimates(target_location);

      for (int id : request.neighbor_ids) {
        node.RecieveNeighborEstimate(
          sensor_nodes_[id].SendEstimatePacket(target_location)
        );
      }
    }

    estimates = CollectEstimatesFromNodes(target_location);
    estimate_log.estimates.push_back(estimates);
  }

  for (SimNode node : passive_nodes) {
    SimNode::RequestPacket request = node.RequestEstimateFromNearestNeighbor(
      target_location
    );

    node.AcceptNeighborEstimate(
      sensor_nodes_[request.neighbor_ids.front()].SendEstimatePacket(
        target_location
      )
    );
  }

  estimates = CollectEstimatesFromNodes(target_location);
  estimate_log.estimates.push_back(estimates);

  return estimate_log;
}

std::vector<SimNode>
SimNodeNetwork::FindSubSetOfNodes(
    const std::map<int, SimNode>& sensor_nodes,
    const SimNode::CoordinatePair& target_location,
    const std::function<bool(
      const SimNode&,
      const SimNode::CoordinatePair&
    )>& CriteriaIsMet) {

  std::vector<SimNode> subset(sensor_nodes.size());

  for (auto entry : sensor_nodes) {
    SimNode node = entry.second;

    if (CriteriaIsMet(node, target_location)) {
      subset.push_back(node);
    }
  }

  return subset;
}

std::vector<double>
SimNodeNetwork::CollectEstimatesFromNodes(
  const SimNode::CoordinatePair& target_location) {

  std::vector<double> estimates(sensor_nodes_.size());

  for (auto entry : sensor_nodes_) {
    SimNode node = entry.second;
    estimates[node.id()] = (node.SendEstimatePacket(target_location)).estimate;
  }

  return estimates;
}

double
SimNodeNetwork::ComputeAverageReading(
  const std::vector<SimNode> estimating_nodes,
  const SimNode::CoordinatePair& target_location) {

  double average_reading = 0.0;
  
  for (SimNode node : estimating_nodes) {
    average_reading += (node.SendEstimatePacket(target_location)).estimate;
  }
  average_reading /= estimating_nodes.size();                        ///////////// TODO: make flexible for weighted average

  return average_reading;
}

bool
SimNodeNetwork::EstimatesHaveSufficientlyConverged(
  const std::vector<double>& estimates,
  const double average_reading) const {

  for (double estimate : estimates) {
    if (estimate != NO_READING) {
      if (fabs(estimate - average_reading) >= error_threshold_) {
        return false;
      }
    }
  }

  return true;
}

bool SimNodeNetwork::EstimatesHaveNotSufficientlyConverged(
  const std::vector<double>& estimates,
  const double average_reading) const {

  return !EstimatesHaveSufficientlyConverged(estimates, average_reading);
}


#endif //_SIM_NODE_NETWORK_CPP_