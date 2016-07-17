/**
 * Get some sample tasks going
 * Thinking that a practice task is a good idea
 * Maybe it could indicate where the relevant bookmark in the task environment is
 * Determine how to go about timing everything
 * Thinking that just timing from start-up/task finishes is fine to help evaluate navigability
 * Add introduction screen (might use that intro/tutorial library for this)
 * Add survey screen (might use that one library for this)
 * JSON schema:
 * {
 * quantitative: {timings}
 * qualitative: {stuff...}
 * }
 * <p/>
 * Later:
 * Confirm when the task is ready to be completed/given up on with a popup or something?
 * Alter the layout to confirm when the user is ready to start the task?
 */

package edu.unr.hci.task;

import android.content.Intent;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.ContactsContract;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;
import android.support.v4.app.FragmentPagerAdapter;
import android.support.v4.view.ViewPager;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;

public class TaskTimingActivity extends AppCompatActivity {
    public static final String TAG = TaskTimingActivity.class.toString();

    public static final long GIVE_UP_TIME = 1000 * 60 * 3; // 3 minutes in milliseconds

    public static final String DATA_DIR = "HCI-Task-Results";
    public static final String DATA_FILE = "experiment_data.txt";

    public static final float DISABLED_ALPHA = 0.4f;

    public static final int TOTAL_ROUNDS = 3;
    public static final int N_TRAINING_TASKS = 3;
    public static final int N_FIRST_ROUND_TASKS = 3;
    public static final int N_SECOND_ROUND_TASKS = 2;

    Intent mActivatingIntent;

    private ViewPager mTaskViewPager;
    private SectionsPagerAdapter mSectionsPagerAdapter;
    private Button mTaskFinishedButton;
    private Button mTaskGiveUpButton;

    Integer mTaskRound;
    String mInterfaceVersion;
    Integer mNTasks;
    ArrayList<String> mTaskDescriptionTexts;
    ArrayList<Integer> mTaskImages;
    private int mCurrentTask;
    private long mTaskStartTime;
    private SessionResult mSessionResult;

    @Override
    public void onBackPressed() {
        // do nothing - disable the back button
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_task_timing);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        this.mActivatingIntent = this.getIntent();

        this.mSectionsPagerAdapter = new SectionsPagerAdapter(getSupportFragmentManager());
        this.setupTaskViewPager(this.mSectionsPagerAdapter);
        this.setupFinishedButton();
        this.setupGiveUpButton();

        String participantId = this.mActivatingIntent.getStringExtra(IntentKeys.PARTICIPANT_ID_EXTRA);
        this.mInterfaceVersion = this.mActivatingIntent.getStringExtra(IntentKeys.INTERFACE_VERSION_EXTRA);
        this.mTaskRound = this.mActivatingIntent.getIntExtra(IntentKeys.TASK_ROUND_EXTRA, 0);
        this.mCurrentTask = 0;
        this.setupTaskParameters(this.mTaskRound);

        this.setupNewTask(this.mCurrentTask);
        this.mSessionResult = new SessionResult(participantId);
        this.setupNewTask(this.mCurrentTask);


        try {
            Log.i(TAG, new JSONObject(this.getIntent().getStringExtra(IntentKeys.EXPERIMENT_DATA_EXTRA)).toString(4));
        } catch (JSONException e) {

        }
    }

    public void setupTaskViewPager(SectionsPagerAdapter pagerAdapter) {
        // Set up the ViewPager with the sections adapter.
        this.mTaskViewPager = (ViewPager) findViewById(R.id.view_task_display);
        if (mTaskViewPager != null) {
            this.mTaskViewPager.setAdapter(pagerAdapter);
        }
    }

    public void setupFinishedButton() {
        this.mTaskFinishedButton = (Button) findViewById(R.id.button_task_complete);
        this.mTaskFinishedButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.i(TAG, "Task Complete.");
                completeTask(true);
            }
        });
    }

    public void setupGiveUpButton() {
        this.mTaskGiveUpButton = (Button) findViewById(R.id.button_task_give_up);
        this.mTaskGiveUpButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                completeTask(false);
            }
        });
    }

    public void setupTaskParameters(int round) {
        switch (round) {
            case 0:
                this.mTaskDescriptionTexts = TRAINING_TASK_TEXTS;
                this.mTaskImages = TRAINING_TASK_IMAGES;
                this.mNTasks = this.mTaskDescriptionTexts.size();
                break;
            case 1:
                this.mTaskDescriptionTexts = TASK_SET_1_TEXTS;
                this.mTaskImages = TASK_SET_1_SCREENSHOTS;
                this.mNTasks = this.mTaskDescriptionTexts.size();
                break;
            case 2:
                this.mTaskDescriptionTexts = TASK_SET_2_TEXTS;
                this.mTaskImages = TASK_SET_2_SCREENSHOTS;
                this.mNTasks = this.mTaskDescriptionTexts.size();
                break;
            default:
                throw new IllegalStateException("This app only uses 3 rounds.");
        }
    }

    public void completeTask(boolean taskFinished) {
        this.storeTaskResult(taskFinished);

        this.mCurrentTask++;

        if (this.mCurrentTask < this.mNTasks) {
            this.setupNewTask(this.mCurrentTask);
        } else {
            this.proceedToNextTransition();
        }
    }

    public void storeTaskResult(boolean taskFinished) {
        long taskDuration = SystemClock.elapsedRealtime() - this.mTaskStartTime;

        SessionResult.CompletionStatus status =
                taskFinished ? SessionResult.CompletionStatus.COMPLETE :
                        SessionResult.CompletionStatus.GIVE_UP;

        this.mSessionResult.addTaskResult(taskDuration, status);
    }

    public void setupNewTask(int taskNumber) {
        String taskDescription = this.mTaskDescriptionTexts.get(taskNumber);
        int screenshotId = this.mTaskImages.get(taskNumber);
        this.mSectionsPagerAdapter.setupNewTask(taskDescription, screenshotId);

        this.mTaskGiveUpButton.setAlpha(DISABLED_ALPHA);
        this.mTaskGiveUpButton.setEnabled(false);
        this.mTaskGiveUpButton.postDelayed(new Runnable() {
                                               @Override
                                               public void run() {
                                                   mTaskGiveUpButton.setEnabled(true);
                                                   mTaskGiveUpButton.setAlpha(1.0f);
                                               }
                                           },
                GIVE_UP_TIME);

        this.mTaskFinishedButton.setEnabled(false);
        this.mTaskFinishedButton.postDelayed(new Runnable() {
                                               @Override
                                               public void run() {
                                                   mTaskFinishedButton.setEnabled(true);
                                                   mTaskFinishedButton.setAlpha(1.0f);
                                               }
                                           },
                1000);

        this.mTaskStartTime = SystemClock.elapsedRealtime();
    }

    public void saveTaskData() {
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {
            File directory = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), DATA_DIR);

            try {
                if (!directory.exists()) {
                    directory.mkdirs();
                }

                File file = new File(directory, String.format(Locale.US, "%s.json", this.mSessionResult.participant));
                if (!file.exists()) {
                    file.createNewFile();
                } else {
                    BufferedReader br = new BufferedReader(new FileReader(file));
                    StringBuilder builder = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) {
                        builder.append(line).append("\n");
                    }
                    br.close();
                }

                BufferedWriter bw = new BufferedWriter(new FileWriter(file));

                bw.write(this.mSessionResult.toJsonString());
                bw.close();
            } catch (Exception e) {
                Log.e(TAG, e.getMessage(), e);
            }
        } else {
            Log.wtf(TAG, "No external storage is mounted - unable to write data to file");
        }
    }

    public void proceedToNextTransition() {
        String transitionMessage;
        switch (this.mTaskRound) {
            case 0:
                transitionMessage = "Ok, now you are ready to try out some NRDC tasks!";

                break;
            case 1:
                transitionMessage = "Take a break and let the facilitator know that you have " +
                        "finished this set of tasks. You can hit the button to proceed after " +
                        "they have given you further instructions.";
                break;
            case 2:
                transitionMessage = "That concludes the task portion of our experiment! Ask your " +
                        "facilitator what to do next.";
                break;
            default:
                throw new IllegalStateException("Attempting to proceed to transition from invalid " +
                        "round of tasks. (" + this.mTaskRound + ")");
        }


        JSONObject experimentData = new JSONObject();
        try {
            experimentData = new JSONObject(this.getIntent().getStringExtra(IntentKeys.EXPERIMENT_DATA_EXTRA));
            JSONArray times = new JSONArray(this.mSessionResult.taskCompletionTimes);
            experimentData.put(DataKeys.TIMINGS + this.mTaskRound, times);
            JSONArray outcomes = new JSONArray(this.mSessionResult.taskPerserveraceResults);
            experimentData.put(DataKeys.OUTCOMES + this.mTaskRound, outcomes);

            Log.i(TAG, experimentData.toString(4));

        } catch (JSONException e) {
            Log.e(TAG, e.getMessage(), e);
        }



        this.mTaskRound++;

        Intent intent;
        if (this.mTaskRound < TOTAL_ROUNDS) {
            intent = new Intent(this, TransitionActivity.class);
            intent.putExtras(this.getIntent());

            intent.removeExtra(IntentKeys.EXPERIMENT_DATA_EXTRA);
            intent.putExtra(IntentKeys.EXPERIMENT_DATA_EXTRA, experimentData.toString());
            intent.putExtra(TransitionActivity.TRANSITION_MESSAGE_EXTRA, transitionMessage);
            intent.putExtra(IntentKeys.TASK_ROUND_EXTRA, this.mTaskRound);
        } else {
            saveExperimentToFile(experimentData);
            intent = new Intent(this, ThankYouActivity.class);
        }

        this.finish();
        startActivity(intent);
    }

    public void saveExperimentToFile(JSONObject sessionData) {
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {
            File directory = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), DATA_DIR);

            try {
//                if (!directory.exists()) {
//                    directory.mkdirs();
//                }
//
//                JSONArray trials;
//                File file = new File(directory, DATA_FILE);
//                if (!file.exists()) {
//                    file.createNewFile();
//                    trials = new JSONArray();
//                } else {
//                    BufferedReader br = new BufferedReader(new FileReader(file));
//                    StringBuilder builder = new StringBuilder();
//                    String line;
//                    while ((line = br.readLine()) != null) {
//                        builder.append(line).append("\n");
//                    }
//                    br.close();
//                    trials = new JSONArray(builder.toString());
//                }

                ensureDataFileExists();
                JSONArray data = getOldExperimentData(DATA_DIR, DATA_FILE);
                data.put(sessionData);
                writeDataToFile(DATA_DIR, DATA_FILE, data);

//                BufferedWriter bw = new BufferedWriter(new FileWriter(file, false));
//
//                bw.write(trials.toString(4));
//                bw.close();
            } catch (Exception e) {
                Log.e(TAG, e.getMessage(), e);
            }
        } else {
            Log.wtf(TAG, "No external storage is mounted - unable to write data to file");
        }
    }

    public void ensureDataFileExists() {
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {
            File directory = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), DATA_DIR);
            try {
                if (!directory.exists()) {
                    directory.mkdirs();
                }

                File file = new File(directory, DATA_FILE);
                if (!file.exists()) {
                    file.createNewFile();
                }
            } catch (IOException e) {
                Log.i(TAG, e.getMessage(), e);
            }
        } else {
            Log.wtf(TAG, "No external storage is mounted - unable to write data to file");
        }
    }

    public JSONArray getOldExperimentData(String dir, String fileName) {
        JSONArray experimentData = null;
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {
            File directory = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), dir);
            File file = new File(directory, fileName);

            try {
                BufferedReader br = new BufferedReader(new FileReader(file));
                StringBuilder builder = new StringBuilder();
                String line;
                while ((line = br.readLine()) != null) {
                    builder.append(line).append("\n");
                }
                br.close();
                experimentData = new JSONArray(builder.toString());
            } catch (IOException|JSONException e) {
                Log.i(TAG, e.getMessage(), e);
            }
        } else {
            Log.wtf(TAG, "No external storage is mounted - unable to write data to file");
        }

        if (experimentData == null) {
            experimentData = new JSONArray();
        }

        return experimentData;
    }

    public void writeDataToFile(String dir, String fileName, JSONArray data) {
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {
            File directory = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS), dir);
            File file = new File(directory, fileName);
            try {
                BufferedWriter bw = new BufferedWriter(new FileWriter(file, false));
                bw.write(data.toString(4));
                bw.close();
            } catch (IOException|JSONException e) {
                Log.i(TAG, e.getMessage(), e);
            }
        } else {
            Log.wtf(TAG, "No external storage is mounted - unable to write data to file");
        }
    }


    public class SectionsPagerAdapter extends FragmentPagerAdapter {

        public static final int N_TABS = 2; // only the description page and the screenshot page

        FragmentManager mFragmentManager;
        TaskTextFragment mTaskDisplayFragment;
        TaskExampleFragment mTaskExampleFragment;

        public SectionsPagerAdapter(FragmentManager fm) {
            super(fm);
            this.mFragmentManager = fm;

            this.mTaskDisplayFragment = new TaskTextFragment();
//            Bundle bundle = new Bundle();
//            bundle.putString(TaskTextFragment.TASK_DESCRIPTION, "hello");
//            this.mTaskDisplayFragment.setArguments(bundle);

            this.mTaskExampleFragment = new TaskExampleFragment();
        }

        @Override
        public Fragment getItem(int position) {
            if (position == 0) {
                return this.mTaskDisplayFragment;
            } else /*if (position == 1)*/ {
                return this.mTaskExampleFragment;
            }
        }

        @Override
        public int getItemPosition(Object object) {
            if (object != null) {
                ((IUpdateableFragment) object).update();
            }

            return super.getItemPosition(object);
        }

        @Override
        public int getCount() {
            return N_TABS;
        }

        public void setupNewTask(String description, int screenshot) {

            this.mTaskDisplayFragment.setDescription(description);
            this.mTaskExampleFragment.setScreenshotId(screenshot);

            this.notifyDataSetChanged();
        }
    }


    public static final ArrayList<String> TRAINING_TASK_TEXTS = new ArrayList<>(Arrays.asList(
            "Your first training task will be to press the green \"Finished!\" button. This is the " +
                    "button you should press when you have completed a task.",
            "Your next task will be to try out swiping between a screenshot and an image. Go on, " +
                    "swipe this text to the left, and swipe right to bring it back. Click the green " +
                    "button when you are done.",
            "Finally, there is a red button that is currently disabled, but will be available 3 " +
                    "minutes after beginning each task. This button is the \"Give Up\" button - you " +
                    "would press this button when you feel stuck and would normally ask someone " +
                    "for help. Click either button to complete this training task."
    ));

    public static final ArrayList<Integer> TRAINING_TASK_IMAGES = new ArrayList<>(Arrays.asList(
            R.drawable.green_thumbs_up,
            R.drawable.green_thumbs_up,
            R.drawable.lets_do_this
    ));

    public static final ArrayList<String> TASK_SET_1_TEXTS = new ArrayList<>(Arrays.asList(
            "Find the current conditions for the \"Sheep Range Mojave Desert Shrub\" location. " +
                    "This should feature temperature and wind readings, along with webcam images from the area.",
            "Use the Geospatial Data Search service to download the permittivity of the " +
                    "soil for the Sheep Range Montane " +
                    "location. Get the data for 10- and 60-minute intervals.",
            "Suggest a data set to the NRDC. For the name field use \"NRDC Test\"; for the email " +
                    "field use \"nrdc@test.com\", and in the description put \"test\".",
            "Use the Webcam Image Archive service to obtain a time-lapse video of the Snake Range " +
                    "East Subalpine location. Collect images from the dates May 1st through 3rd 2016, " +
                    "use all times of day, collect the images from the southeast facing camera, " +
                    "and convert the images into a video series. Watch the resulting video.",
            "Use the Geospatial Data Search service to download the snowfall sensor " +
                    "temperature data (in degrees Fahrenheit) for the Snake Range East Sagebrush " +
                    "location. Get the data for 1-minute intervals.",
            "Locate the Image Gallery. View enlarged versions of the Fly Ranch Geyser (a colorful " +
                    "geyser in mid eruption). Then view an enlarged photo of a colorful sunset " +
                    "over a snow covered hill.",
            "Locate a description of what DataONE is."
    ));

    public static final ArrayList<Integer> TASK_SET_1_SCREENSHOTS = new ArrayList<>(Arrays.asList(
            R.drawable.current_conditions_sheep_range_mojave_desert_shrub,
            R.drawable.geospatial_sheep_range_montane,
            R.drawable.suggest_dataset,
            R.drawable.webcam_image_archive_snake_range_east_subalpine,
            R.drawable.geospatial_snake_range_east_sagebrush,
            R.drawable.image_gallery_fly_ranch_geyser,
            R.drawable.data_one
    ));

    public static final ArrayList<String> TASK_SET_2_TEXTS = new ArrayList<>(Arrays.asList(
            "Find the current conditions for the \"Snake Range West Pinyon-Juniper\" location. " +
                    "This should feature temperature and wind readings, along with webcam images from the area.",
            "Use the Geospatial Data Search service to download the tree trunk radius " +
                    "growth for the Snake Range West Subalpine " +
                    "location. Get the data for 30- and 60-minute intervals.",
            "Report some research results to the NRDC. For the name field use \"NRDC Test\"; for the email " +
                    "field use \"nrdc@test.com\", and in the description put \"test\".",
            "Use the Webcam Image Archive service to obtain a time-lapse video of the Sheep Range " +
                    "Blackbrush location. Collect images from the dates May 1st through 3rd 2016, " +
                    "use all times of day, collect the images from the southeast facing camera, " +
                    "and convert the images into a video series. Watch the resulting video.",
            "Use the Geospatial Data Search service to download the solar radiation " +
                    "data in kW/square meter for the Rockland Summit " +
                    "location. Get only the averages.",
            "Find the \"About Us\" information that states the NRDC mission statement",
            "Locate a description of what CUAHSI is."

    ));

    public static final ArrayList<Integer> TASK_SET_2_SCREENSHOTS = new ArrayList<>(Arrays.asList(
            R.drawable.current_conditions_snake_range_pinyon_juniper,
            R.drawable.geospatial_snake_range_west_subalpine,
            R.drawable.submit_research_results ,
            R.drawable.webcam_image_archive_sheep_range_blackbrush,
            R.drawable.geospatial_rockland_summit,
            R.drawable.about_us_nrdc,
            R.drawable.connections_cuahsi
            ));

    public ArrayList<String> taskDescriptions = new ArrayList<>(Arrays.asList(
            "Use the Geospatial Data Search to find the mean tree radial growth in the Snake Range East Sagebrush location.",
            "Vinh help me come up with tasks!",
            "There's darkness everywhere Ryan. You just can't see it because the sun is such an attention whore."
    ));
    public ArrayList<String> taskScreenshots = new ArrayList<>(Arrays.asList(
            "@drawable/geospatial_task",
            "@drawable/web_image_archive_task",
            "@drawable/wilfred"
    ));
}
