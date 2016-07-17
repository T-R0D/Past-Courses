//package edu.unr.hci.task.instructions;
//
//import android.content.Intent;
//import android.support.v7.app.AppCompatActivity;
//import android.os.Bundle;
//
//import com.chyrta.onboarder.OnboarderActivity;
//import com.chyrta.onboarder.OnboarderPage;
//
//import java.util.ArrayList;
//import java.util.List;
//
//import edu.unr.hci.task.R;
//import edu.unr.hci.task.TaskTimingActivity;
//
//public class InstructionsActivity extends OnboarderActivity {
//
//    List<OnboarderPage> onboarderPages;
//
//    String mParticipantId;
//
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//
//        mParticipantId = getIntent().getStringExtra(TaskTimingActivity.PARTICIPANT_EXTRA);
//
////        onboarderPages = new ArrayList<OnboarderPage>();
//
//        // Create your first page
////        OnboarderPage onboarderPage1 = new OnboarderPage("Title 1", "Description 1");
////        OnboarderPage onboarderPage2 = new OnboarderPage(R.string.app_name, R.string.app_description, R.drawable.wilfred);
//
//        // You can define title and description colors (by default white)
////        onboarderPage1.setTitleColor(R.color.black);
////        onboarderPage1.setDescriptionColor(R.color.white);
//
//        // Don't forget to set background color for your page
////        onboarderPage1.setBackgroundColor(R.color.my_awesome_color);
//
//        // Add your pages to the list
////        onboarderPages.add(onboarderPage1);
////        onboarderPages.add(onboarderPage2);
//
//        // And pass your pages to 'setOnboardPagesReady' method
//        setOnboardPagesReady(onboarderPages);
//
//    }
//
//    @Override
//    public void onSkipButtonPressed() {
//        // Define your actions when the user press 'Skip' button
//    }
//
//    @Override
//    public void onFinishButtonPressed() {
//        // Define your actions when the user press 'Finish' button
//        Intent intent = new Intent(this, TaskTimingActivity.class);
//        intent.putExtra(TaskTimingActivity.PARTICIPANT_EXTRA, this.mParticipantId);
//        this.finish();
//        startActivity(intent);
//    }
//}
