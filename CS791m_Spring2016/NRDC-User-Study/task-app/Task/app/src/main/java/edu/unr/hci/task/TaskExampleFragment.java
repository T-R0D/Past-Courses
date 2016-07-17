package edu.unr.hci.task;

import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;

public class TaskExampleFragment extends Fragment implements IUpdateableFragment {

    public static final String SCREENSHOT_NAME = "SCREENSHOT_NAME";


    ImageView mScreenshotView;
    int mScreenshotId;

    public TaskExampleFragment() {
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View rootView = inflater.inflate(R.layout.fragment_task_example, container, false);
        this.mScreenshotView = (ImageView) rootView.findViewById(R.id.imageview_screenshot);
//        this.mScreenshotId = R.drawable.wilfred;
        this.mScreenshotView.setImageResource(this.mScreenshotId);

        return rootView;
    }

    public void setScreenshotId(int screenshotId) {
        this.mScreenshotId = screenshotId;
    }

    @Override
    public void update() {
        this.mScreenshotView.setImageResource(this.mScreenshotId);
    }
}
