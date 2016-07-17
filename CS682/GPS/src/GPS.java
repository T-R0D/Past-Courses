import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Terence
 * on 8/30/2014.
 */
public class GPS {

    static List<Condition> currentState;
    static List<Operation> currentOperations;

    public static void main(String[] args) {
        currentState = new ArrayList<>(Arrays.asList(
                Condition.SON_AT_HOME,
                Condition.CAR_NEEDS_BATTERY,
                Condition.HAVE_MONEY,
                Condition.HAVE_PHONE_BOOK
        ));

        currentOperations = new ArrayList<>(Arrays.asList(
                new Operation(
                        "Driving son to school...",
                        Arrays.asList(Condition.SON_AT_HOME, Condition.CAR_WORKS),
                        Arrays.asList(Condition.SON_AT_SCHOOL),
                        Arrays.asList(Condition.SON_AT_HOME)
                ),
                new Operation(
                        "Shop installing car battery...",
                        Arrays.asList(Condition.CAR_NEEDS_BATTERY, Condition.SHOP_KNOWS_PROBLEM, Condition.SHOP_HAS_MONEY),
                        Arrays.asList(Condition.CAR_WORKS),
                        Arrays.asList()
                ),
                new Operation(
                        "Telling shop problem...",
                        Arrays.asList(Condition.IN_COMMUNICATION_WITH_SHOP),
                        Arrays.asList(Condition.SHOP_KNOWS_PROBLEM),
                        Arrays.asList()
                ),
                new Operation(
                        "Calling shop on phone...",
                        Arrays.asList(Condition.KNOW_PHONE_NUMBER),
                        Arrays.asList(Condition.IN_COMMUNICATION_WITH_SHOP),
                        Arrays.asList()
                ),
                new Operation(
                        "Looking up shop's phone number...",
                        Arrays.asList(Condition.HAVE_PHONE_BOOK),
                        Arrays.asList(Condition.KNOW_PHONE_NUMBER),
                        Arrays.asList()
                ),
                new Operation(
                        "Giving shop money...",
                        Arrays.asList(Condition.HAVE_MONEY),
                        Arrays.asList(Condition.SHOP_HAS_MONEY),
                        Arrays.asList(Condition.HAVE_MONEY)
                )
        ));

        GrandProblemSolve(currentState, new ArrayList<Condition>(Arrays.asList(Condition.SON_AT_SCHOOL)), currentOperations);
    }

    public static boolean GrandProblemSolve(List<Condition> pState, List<Condition> pGoals, List<Operation> pAvailableOperations) {
        for(Condition goal : pGoals) {
            if(!goalIsAcheived(goal, pState)) {
                return false;
            }
        }

        return true;
    }

    public static boolean goalIsAcheived(Condition pGoal, List<Condition> pCurrentState) {
        for(Condition condition : pCurrentState) {
            if(condition.equals(pGoal)) {
                return true;
            }
        }

        List<Operation> candidateOperations = findAll(pGoal, currentOperations);

        for(Operation candidate : candidateOperations) {
            if(applyingOperationAcheivesGoal(candidate, pCurrentState)) {
                return true;
            }
        }

        return false;
    }

    public static boolean applyingOperationAcheivesGoal(Operation pOperation, List<Condition> pCurrentState) {
        for(Condition precondition : pOperation.preconditions()) {
            if(!goalIsAcheived(precondition, pCurrentState)) {
                return false;
            }
        }

        // apply the operation
        System.out.println("Executing: " + pOperation.name());
        pCurrentState.removeAll(pOperation.deleteList());
        for(Condition condition : pOperation.addList()) {
            if(!pCurrentState.contains(condition)) {
                pCurrentState.add(condition);
            }
        }

        return true;
    }

    public static List<Operation> findAll(Condition pGoal, List<Operation> pAvailableOperations) {
        List<Operation> result = new ArrayList<>();

        for(Operation operation : pAvailableOperations) {
            if(operation.isAppropriateToAcheiveGoal(pGoal)) {
                result.add(operation);
            }
        }

        return result;
    }

}
