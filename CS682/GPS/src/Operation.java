import java.util.List;

/**
 * Created by Terence
 * on 8/31/2014.
 */
public class Operation {
    private String mName;
    private List<Condition> mPreconditions;
    private List<Condition> mAddList;
    private List<Condition> mDeleteList;

    public Operation(String pName, List<Condition> pPreconditions, List<Condition> pAddList, List<Condition> pDeleteList) {
        mName = pName;
        mPreconditions = pPreconditions;
        mAddList = pAddList;
        mDeleteList = pDeleteList;
    }

    public String name() {
        return mName;
    };

    public List<Condition> preconditions() {
        return mPreconditions;
    }

    public List<Condition> addList() {
        return mAddList;
    }

    public List<Condition> deleteList() {
        return mDeleteList;
    }

    public boolean isAppropriateToAcheiveGoal(Condition pGoal) {
        for(Condition result : mAddList) {
            if(result.equals(pGoal)) {
                return true;
            }
        }

        return false;
    }
}