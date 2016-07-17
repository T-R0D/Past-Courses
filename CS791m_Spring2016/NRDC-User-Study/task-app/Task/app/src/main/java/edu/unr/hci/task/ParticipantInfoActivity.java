package edu.unr.hci.task;

import android.content.Intent;
import android.provider.ContactsContract;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.NumberPicker;
import android.widget.RadioGroup;

import org.json.JSONException;
import org.json.JSONObject;


public class ParticipantInfoActivity extends AppCompatActivity {

    public static final String TAG = ParticipantInfoActivity.class.toString();

    public static final String INTERFACE_A = "INTERFACE_A";
    public static final String INTERFACE_1 = "INTERFACE_1";

    public static final int LOWEST_PARTICIPANT = 0;
    public static final int HIGHEST_PARTICIPANT = 99;

    Integer mParticipantId;
    String mInterfaceVersion;
    NumberPicker mParticipantIdPicker;
    RadioGroup mInterfaceVersionGroup;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_participant_info);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        this.setupParticipantIdPicker();
        this.setupInterfaceVersionRadioButtons();
        this.setupBeginButton();
    }

    public void setupParticipantIdPicker() {
        this.mParticipantIdPicker = (NumberPicker) findViewById(R.id.picker_participant_id);
        if (this.mParticipantIdPicker != null) {
            this.mParticipantIdPicker.setMinValue(LOWEST_PARTICIPANT);
            this.mParticipantIdPicker.setMaxValue(HIGHEST_PARTICIPANT);
            this.mParticipantIdPicker.setWrapSelectorWheel(false);

            this.mParticipantIdPicker.setOnValueChangedListener(new NumberPicker.OnValueChangeListener() {
                @Override
                public void onValueChange(NumberPicker picker, int oldVal, int newVal) {
                    mParticipantId = newVal;
                }
            });
        }
    }

    public void setupInterfaceVersionRadioButtons() {
        this.mInterfaceVersionGroup = (RadioGroup) findViewById(R.id.radio_group_interface_version);
    }

    public void setupBeginButton() {
        Button beginButton = (Button) findViewById(R.id.button_begin);
        if (beginButton != null) {
            beginButton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    try {
                        beginExperiment();
                    } catch (JSONException e) {
                        Log.e(TAG, e.getMessage(), e);
                    }
                }
            });
        }
    }

    public void beginExperiment() throws JSONException {
        int pickedInterfaceId;
        pickedInterfaceId = this.mInterfaceVersionGroup.getCheckedRadioButtonId();

        if (pickedInterfaceId == R.id.radio_nrdc_a) {
            mInterfaceVersion = INTERFACE_A;
        } else if (pickedInterfaceId == R.id.radio_nrdc_1) {
            mInterfaceVersion = INTERFACE_1;
        }

        JSONObject experimentData = new JSONObject();
        try {
            experimentData.put(DataKeys.PARTICIPANT_ID, this.mParticipantIdPicker.getValue());
            experimentData.put(DataKeys.INTERFACE_VERSION, mInterfaceVersion);
        } catch (JSONException e) {
            Log.e(TAG, e.getMessage(), e);
        }

        Intent intent = new Intent(this, TransitionActivity.class);
        intent.putExtra(IntentKeys.TASK_ROUND_EXTRA, 0);
        intent.putExtra(TransitionActivity.TRANSITION_MESSAGE_EXTRA, "Here's a training exercise! " +
                "The following task(s) will help familiarize you with this timekeeping app.");
        intent.putExtra(IntentKeys.EXPERIMENT_DATA_EXTRA, experimentData.toString());
        intent.putExtra(IntentKeys.PARTICIPANT_ID_EXTRA, this.mParticipantIdPicker.getValue());
        intent.putExtra(IntentKeys.INTERFACE_VERSION_EXTRA, mInterfaceVersion);
        this.finish();
        startActivity(intent);
    }
}
