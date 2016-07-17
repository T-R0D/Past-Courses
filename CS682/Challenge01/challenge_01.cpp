#include <cstdio>
#include <iostream>
#include <functional>
#include <string>
#include <vector>
#include <list>
#include <queue>

class Person {
 public:
 	enum class PersonType : bool {
 		MISSIONARY,
 		CANNIBAL
 	};

 	Person(const std::string& pName, const PersonType& pType) :
 			mName{pName}, mType{pType} {
 	}

 	Person(const Person& pThat) {
 		if (this != &pThat) {
 			mName = pThat.name();
 			mType = pThat.type();
 		}
 	}

 	std::string name() const {
 		return mName;
 	}

 	PersonType type() const {
 		return mType;
 	}

 	bool operator==(const Person& pThat) {
 		if (this != &pThat) {
 			if (this->mName != pThat.mName || this->mType != pThat.mType) {
 				return false;
 			}
 		}

 		return true;
 	}

 	bool operator!=(const Person& pThat) {
 		return *this == pThat;
 	}

 	static std::string toString(const PersonType& pType) {
 		switch (pType) {
 			case PersonType::MISSIONARY:
 				return "MISSIONARY";
 				break;
 			case PersonType::CANNIBAL:
 				return "CANNIBAL";
 				break;
 			default:
 				return "UNKNOWN";
 				break;
 		}
 	}

 private:
 	std::string mName;
 	PersonType mType;
};


class ProblemState {
 public:
	enum class Location : char {
 		START,
 		BOAT,
 		GOAL
 	};

 	ProblemState(
 		const int pNumCannibals,
 		const int pNumMissionaries,
 		const int pBoatSize,
 		std::vector<Person> pStartSideState,
 		std::function<bool(std::vector<Person>&)> pStartSideIsValid,
 		std::vector<Person> pBoatState,
 		std::function<bool(std::vector<Person>&)> pBoatIsValid,
 		std::vector<Person> pGoalSideState,
 		std::function<bool(std::vector<Person>&)> pGoalSideIsValid) :
	 		mNumHumans{pNumCannibals + pNumMissionaries},
 			mBoatMaxOccupancy{pBoatSize},
 			mStartingSide{pStartSideState},
 			mStartingSideIsValid{pStartSideIsValid},
 			mBoat{pBoatState},
 			mBoatIsValid{pBoatIsValid},
 			mGoalSide{pGoalSideState},
 			mGoalSideIsValid{pGoalSideIsValid} {
 	}

 	ProblemState(
 		const ProblemState& pOldState,
 		const std::vector<Person>& pStartSide,
 		const std::vector<Person>& pBoat,
 		const std::vector<Person>& pGoalSide) {
 		*this = pOldState;
 		this->mStartingSide = pStartSide;
 		this->mBoat = pBoat;
 		this->mGoalSide = pGoalSide;
 	}


 	ProblemState(const ProblemState& pThat) {
 		*this = pThat;
 	}

 	~ProblemState() {
 	}

 	ProblemState& operator=(const ProblemState& pThat) {
 		if (this != &pThat) {
 			mNumHumans = pThat.mNumHumans;
 			mBoatMaxOccupancy = pThat.mBoatMaxOccupancy;
 			mStartingSide = pThat.mStartingSide;
 			mGoalSide = pThat.mGoalSide;
 			mBoat = pThat.mBoat;
 			mStartingSideIsValid = pThat.mStartingSideIsValid;
 		}

 		return *this;
 	}


 	std::vector<ProblemState> generateAdjacentStates() {
 		std::vector<ProblemState> adjacentStates;

 		if (mStartingSide.size() < mNumHumans) {
 			for (auto person : mBoat) {
 				std::vector<Person> startSide = mStartingSide;
 				std::vector<Person> boat = mBoat;
 				std::vector<Person> goalSide = mGoalSide;

 				for (auto person_ : mBoat) {
 					if (person_ != person) {
 						boat.push_back(person_);
 					}
 				}

 				startSide.push_back(person);

 				ProblemState neighborState(*this, startSide, boat, goalSide);

 				if (neighborState.isValid()) {
 					adjacentStates.push_back(neighborState);
 				}
 			}
 		}

 		// if (mBoat.size() < mBoatMaxOccupancy) {

 		// }

 		// if (mGoalSide.size() < mNumHumans) {

 		// }

 		return adjacentStates;
 	}


	bool isValid() const {
		return true;
	}

 	std::string toString() const {
 		std::string returnVal;

 		returnVal += "People on the starting side:\n";
 		for (auto person : mStartingSide) {
 			returnVal += person.name() + "\n";
 		}

 		returnVal += "\nPeople in the boat:\n";
 		for (auto person : mBoat) {
 			returnVal += person.name() + "\n";
 		}

 		returnVal += "\nPeople on the goal side:\n";
 		for (auto person : mGoalSide) {
 			returnVal += person.name() + "\n";
 		}
 		returnVal += "\n";

 		return returnVal;
 	}

 private:
 	int mNumHumans;
 	int mBoatMaxOccupancy;
 	bool mBoatIsOnGoalSide;
 	std::vector<Person> mStartingSide;
 	std::vector<Person> mGoalSide;
 	std::vector<Person> mBoat;
 	std::function<bool(std::vector<Person>&)> mStartingSideIsValid;
 	std::function<bool(std::vector<Person>&)> mBoatIsValid;
 	std::function<bool(std::vector<Person>&)> mGoalSideIsValid;
 	std::list<std::string> mPreviousMoves;
};


int main(int argc, char** argv) {

// std::function<bool(std::vector<Person>&)> fn = [] (std::vector<Person>& pListOfPeople) -> bool {return false;};
// std::vector<Person> test = {{"Cannibal_00", Person::PersonType::CANNIBAL},
// 			{"Cannibal_01", Person::PersonType::CANNIBAL}};


	ProblemState currentState(
		3,
		3,
		2,
		{
			{"Cannibal_00", Person::PersonType::CANNIBAL},
			{"Cannibal_01", Person::PersonType::CANNIBAL},
			{"Cannibal_02", Person::PersonType::CANNIBAL},
			{"Missionary_00", Person::PersonType::MISSIONARY},
			{"Missionary_01", Person::PersonType::MISSIONARY},
			{"Missionary_02", Person::PersonType::MISSIONARY}
		},
		[] (std::vector<Person>& pListOfPeople) -> bool {return false;},
		{},
		[] (std::vector<Person>& pListOfPeople) -> bool {return false;},
		{},
		[] (std::vector<Person>& pListOfPeople) -> bool {return false;}
	);

// 	std::cout << currentState.toString();

// 	printf("blah blah\n");

	return 0;
}
