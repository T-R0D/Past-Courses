package edu.unr.hci.task;

import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

public class TaskTextFragment extends Fragment implements IUpdateableFragment {

    public static final String TAG = TaskTextFragment.class.toString();

    public static final String TASK_DESCRIPTION = "TASK_DESCRIPTION";

    public String mDescription;
    public TextView mTaskTextView;


    public TaskTextFragment() {
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        Log.i(TAG, "onCreateView is being called!");

        View rootView = inflater.inflate(R.layout.fragment_task_display, container, false);

//        this.mDescription = this.getArguments().getString(TASK_DESCRIPTION);

        this.mTaskTextView = (TextView) rootView.findViewById(R.id.text_task_description);
        this.mTaskTextView.setText(this.mDescription);

        return rootView;
    }



    public void setDescription(String description) {
        Log.i(TAG, "in setDescription in Fragment: " + description);

        this.mDescription = description;
    }

    public void update() {
        Log.i(TAG, "update in fragment");

        if (this.mTaskTextView == null) {
            Log.wtf(TAG, "how is this null?");
        }
        this.mTaskTextView.setText(this.mDescription);

    }
}
