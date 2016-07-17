#ifndef __PROBLEMSTATE_HPP__
#define __PROBLEMSTATE_HPP__

#include <vector>
#include <string>

const std::vector< std::vector<int> > PROBLEM_ACTIONS = {
	{2, 0, 1},
	{1, 0, 1},
	{1, 1, 1},
	{0, 1, 1},
	{0, 2, 1}
};

class ProblemState {
 public:
	ProblemState();
	ProblemState(const ProblemState& pThat);
	ProblemState& operator=(const ProblemState& pThat);
	ProblemState applyAction(const std::vector<int>& pAction);

	bool isValid() const;
	bool validVector(const std::vector<int>& pState) const;
	bool isSuccess() const;
	std::string toString() const;
	std::vector<int> complementVector(const std::vector<int>& pVector) const;
	std::string actionToString(const std::vector<int>& pAction) const;

 
	std::vector<int> mState;
	std::vector<std::string> mActions;
};

#endif //__PROBLEMSTATE_HPP__