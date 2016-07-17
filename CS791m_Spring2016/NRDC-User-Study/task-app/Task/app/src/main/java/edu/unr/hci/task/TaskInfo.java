package edu.unr.hci.task;

import java.util.ArrayList;
import java.util.Arrays;

public class TaskInfo {
    public String text;
    public Integer imageId;

    TaskInfo(String text, int imageId) {

    }

    public static final ArrayList<TaskInfo> t = new ArrayList<>(Arrays.asList(
            new TaskInfo("hello", R.drawable.wilfred)
    ));
}
