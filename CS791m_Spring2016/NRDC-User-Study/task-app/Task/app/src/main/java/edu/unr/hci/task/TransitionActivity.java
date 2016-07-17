package edu.unr.hci.task;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

public class TransitionActivity extends AppCompatActivity {

    public static final String TRANSITION_MESSAGE_EXTRA = TransitionActivity.class + "-transition_message_extra";

    Intent mActivatingIntent;
    TextView mTransitionText;
    Button mProceedButton;

    @Override
    public void onBackPressed() {
        // do nothing - disable the back button
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_transition);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        this.mActivatingIntent = this.getIntent();

        this.setupTransitionMessage();
        this.setupProceedButton();
    }

    public void setupTransitionMessage() {
        this.mTransitionText = (TextView) findViewById(R.id.text_transition_message);
        if (this.mTransitionText != null) {
            String transtionMessage = mActivatingIntent.getStringExtra(TRANSITION_MESSAGE_EXTRA);
            this.mTransitionText.setText(transtionMessage);
        }
    }

    public void setupProceedButton() {
        this.mProceedButton = (Button) findViewById(R.id.button_transition_proceed);
        if (this.mProceedButton != null) {
            this.mProceedButton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    proceedToNextActivity(0);
                }
            });
        }
    }

    void proceedToNextActivity(int whichActivity) {
        Intent intent = new Intent(this, TaskTimingActivity.class);
        intent.putExtras(this.mActivatingIntent.getExtras());
        this.finish();
        startActivity(intent);
    }
}
