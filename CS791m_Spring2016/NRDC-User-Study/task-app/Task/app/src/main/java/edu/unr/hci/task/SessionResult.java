package edu.unr.hci.task;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

public class SessionResult {
    public static final String PARTICIPANT = "participant";
    public static final String TIMES = "times";
    public static final String PERSEVARANCE = "perseverance";

    public String participant;
    public List<Long> taskCompletionTimes;
    public List<String> taskPerserveraceResults;

    public SessionResult(String participant) {
        this.participant = participant;
        this.taskCompletionTimes = new ArrayList<>();
        this.taskPerserveraceResults = new ArrayList<>();
    }

    public void addTaskResult(long time, CompletionStatus status) {
        this.taskCompletionTimes.add(time);
        this.taskPerserveraceResults.add(status.name());
    }

    public int tasksCompleted() {
        return this.taskCompletionTimes.size();
    }

    public String toJsonString() {
        JSONObject json = new JSONObject();
        try {
            json.put(PARTICIPANT, this.participant);
            json.put(TIMES, new JSONArray(this.taskCompletionTimes));
            json.put(PERSEVARANCE, new JSONArray(this.taskPerserveraceResults));
        } catch (JSONException e) {
            System.out.println(e.getCause());
        }
        System.out.println(json.toString());

        return String.format("Implement me! %d results stored.", this.taskCompletionTimes.size());
    }

    public enum CompletionStatus{
        COMPLETE,
        GIVE_UP,
    }
}
