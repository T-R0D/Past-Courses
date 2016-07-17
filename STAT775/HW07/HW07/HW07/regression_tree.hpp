#ifndef _REGRESSION_TREE_HPP_
#define _REGRESSION_TREE_HPP_

#include <Eigen\Dense>

#include "random_discriminant.hpp"

struct RegressionTreeNode {
  DiscriminantModel model;
  RegressionTreeNode* branch_0;
  RegressionTreeNode* branch_1;
};

struct RegressionTree {
  RegressionTreeNode* root;
};



RegressionTreeNode*
CreateRegressionTreeNode(
  const unsigned dimensionality,
  const Eigen::MatrixXd* train_data,
  const Eigen::VectorXi labels,
  double sufficient_accuracy) {
/**
 *
 */
  RegressionTreeNode* new_node =
    (RegressionTreeNode*) malloc(sizeof(RegressionTreeNode));

  InitDiscriminantModel(&(new_node->model, dimensionality), 
}


void
BuildRegressionTree(
  RegressionTree* self,
  unsigned dimensionality,
  const Eigen::MatrixXd* train_data,
  const Eigen::VectorXi labels,
  double sufficient_accuracy) {
/**
 *
 */
  self->root = CreateRegressionTreeNode(train_data, labels);
}

RegressionTreeNode*
CreateRegressionTreeNode(
  const Eigen::MatrixXd* train_data,
  const Eigen::VectorXi* labels) {
/**
 *
 */
  
}


#endif //#define _REGRESSION_TREE_HPP_