import javax.sound.sampled.*;
import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.util.*;

/**
 * <h1>LetterGuessingExperiment</h1>
 * 
 * <h3>Summary</h3>
 * 
 * <ul>
 * <li>Experiment software implementing Shannon's letter-guessing experiment
 * </ul>
 * <p>
 * 
 * <h3>Related References</h3>
 * 
 * <ul>
 * <li>
 * <a href="http://languagelog.ldc.upenn.edu/myl/Shannon1950.pdf">Prediction and entropy of printed
 * English</a>, by C. E. Shannon (1951, <i>Bell System Technical Journal, 30</i>, pp. 50-64).  This is the original
 * paper where the letter-guessing experiment is described.
 * <p>
 * 
 * <li>
 * <a href="http://www.yorku.ca/mack/HCIbook/"><i>Human-Computer Interaction: An Empirical Research
 * Perspective</i></a>, by MacKenzie (2013). Additional discussion is given on the experiment &ndash; with 
 * an HCI perspective.  See Section 2.6.1 ("Redundancy in language").
 * <p> 
 * </ul>
 * <p>
 * 
 * <h3>Shannon's Letter-Guessing Experiment</h3>
 * 
 * The following is an excerpt from Shannon's paper, where the experiment is explained:
 * <p>
 * 
 * <center> <img src="LetterGuessingExperiment-1.jpg"> </center>
 * <p>
 * 
 * For the guessing mode described above, the participant is allowed one guess per letter.
 * <p>
 * 
 * Shannon also describes a second guessing mode, where the participant continues to guess until the
 * correct letter is determined:
 * <p>
 * 
 * <center> <img src="LetterGuessingExperiment-2.jpg"> </center>
 * <p>
 * 
 * Both modes are implemented in this software.
 * <p>
 * 
 * <h3>Running the Experiment Software</h3>
 * <p>
 * 
 * <a href="http://www.yorku.ca/mack/HCIbook/Running/">Click here</a> for instructions on
 * launching/running the application.
 * <p>
 * 
 * <h3>Setup Parameters</h3>
 * 
 * Upon launching, the application presents a setup dialog:
 * <p>
 * 
 * <center> <img src="LetterGuessingExperiment-3.jpg"> </center>
 * <p>
 * 
 * The default parameter settings are read from a configuration file called
 * <code>LetterGuessingExperiment.cfg</code>. This file is created automatically when the
 * application is launched for the first time. The default parameter settings may be changed through
 * the setup dialog. The setup parameters are as follows:
 * <p>
 * 
 * <blockquote>
 * <table border="1" cellspacing="0" cellpadding="6" valign="top">
 * <tr bgcolor="#cccccc">
 * <th>Parameter
 * <th>Description
 * 
 * <tr>
 * <td valign="top">Participant Code
 * <td>
 * Identifies the current participant. This is used in forming the names for the output data files.
 * Also, the sd2 output data file includes a column with the participant code.
 * <p>
 * 
 * 
 * <tr>
 * <td valign="top">Condition Code
 * <td>An arbitrary code used to associate a test condition with this invocation. This parameter
 * might be useful if the software is used in an experiment where a condition is not inherently part
 * of the application (e.g., Gender &rarr; male, female). The condition code is used in forming the
 * name for the output data file. Also, the sd2 output data file contains a column with the
 * condition code.
 * <p>
 * 
 * Note: The setup dialog does not include an entry for "Block code". The block code is generated
 * automatically by the software.
 * <p>
 * 
 * <tr>
 * <td valign="top">Number of Phrases
 * <td>Specifies the number of phrases presented to the participant in the current block.
 * <p>
 * 
 * <tr>
 * <td valign="top">Phrases File
 * <td>Specifies the file from which phrases are selected for input. Phrases are drawn from the file
 * at random. Typically, <code><a href="phrases2.txt">phrases2.txt</a></code> is used. This is the
 * phrase set published by MacKenzie and Soukoreff in 2003 (<a
 * href="http://www.yorku.ca/mack/chi03b.html">click here</a>).
 * <p>
 * 
 * <tr>
 * <td valign="top">Beep on Error
 * <td>A checkbox item that, if set, configures the application to output an audible beep when the
 * user makes an incorrect guess.
 * <p>
 * 
 * <tr>
 * <td valign="top">Guessing Mode
 * <td>Set the guessing mode, as per the two modes described by Shannon (see above).
 * <p>
 * </table>
 * </blockquote>
 * <p>
 * 
 * The two guessing modes are as follows:
 * <p>
 * 
 * <h3>Mode 1 - One Guess Per Letter</h3>
 * 
 * Mode 1 is the one-guess mode, where the participant gets one guess per letter. Here's a sample
 * dialog:
 * <p>
 * 
 * <center> <img src="LetterGuessingExperiment-4.jpg"> </center>
 * <p>
 * 
 * The text to guess is presented in the top line. It appears initially as a string of asterisks.
 * The text is revealed letter by letter as guessing proceeds. The text in the second line is what
 * Shannon called the "reduced text". Here, a dash indicates a correct guess, a letter indicates an
 * incorrect guess.
 * <p>
 * 
 * Typical results popup:
 * <p>
 * 
 * <center> <img src="LetterGuessingExperiment-5.jpg"> </center>
 * <p>
 * 
 * KSPC is "keystrokes per character", computed as
 * <p>
 * 
 * <code><pre>
 *      KSPC = (correct + incorrect) / number_of_letters
 *      </code></pre>
 * 
 * For the one-guess mode, KSPC is always 1.00.
 * <p>
 * 
 * <h3>Mode 2 - Guess Until Correct</h3>
 * 
 * Mode 2 is the correct-guess mode. For each letter, the participant continues to guess until she
 * guesses correctly. Here's a sample dialog:
 * <p>
 * 
 * <center> <img src="LetterGuessingExperiment-6.jpg"> </center>
 * <p>
 * 
 * For the correct-guess mode, the count is shown as a single character in the reduced text (second
 * line). Counts above 9 appear as A (10), B (11), C (12), etc. Keys are darkened for incorrect
 * guesses, as a reminder to the participant of letters already visited.
 * <p>
 * 
 * Typical results popup for the correct-guess mode:
 * <p>
 * 
 * <center> <img src="LetterGuessingExperiment-6a.jpg"> </center>
 * <p>
 * 
 * Here, the number of correct guesses is always equal to the number of letters in the presented
 * text. KSPC is very likely > 1, since the participant is likely to guess incorrectly, at least
 * some of the time. (But, who knows, some people are pretty good at this. <a
 * href="http://www.youtube.com/watch?v=dEuWu8NZqts&feature=related">Click here</a> to see an
 * example from Wheel of Fortune.)
 * <p>
 * 
 * <h3>Output Data Files</h3>
 * 
 * Here are some example output data files:
 * <p>
 * 
 * <dl>
 * <dt>One-guess mode:
 * <dd>
 * <ul>
 * <li><a href="LetterGuessingExperiment-mode1-sd1-example.txt">sd1 example</a>,
 * <li><a href="LetterGuessingExperiment-mode1-sd2-example.txt">sd2 example</a>
 * </ul>
 * <p>
 * 
 * <dt>Correct-guess mode:
 * <dd>
 * <ul>
 * <li><a href="LetterGuessingExperiment-mode2-sd1-example.txt">sd1 example</a>,
 * <li><a href="LetterGuessingExperiment-mode2-sd2-example.txt">sd2 example</a></li>
 * </dl>
 * 
 * The reduced text in the sd2 files is the same as appeared on the display during guessing. The
 * reduced text in the sd1 files is slightly different, however. Here, the reduced text shows the
 * actual guesses. For mode 1, this is a single character &mdash; the user's guess. For mode 2, this
 * is a series of characters showing the progression of guesses.
 * <p>
 * 
 * The data in the sd2 files are full-precision, comma-delimited. Importing into a spreadsheet
 * application provides a convenient method to examine the data on a phrase-by-phrase basis. The
 * data in the sd2 example file above (correct-guess mode), are shown below as they might appear
 * after importing into Microsoft <i>Excel</i>: (click to enlarge)
 * <p>
 * 
 * <center><a href="LetterGuessingExperiment-9.jpg"><img src="LetterGuessingExperiment-9.jpg"
 * width=800></a> </center>
 * <p>
 * 
 * Actual output files use "LetterGuessingExperiment" as the base filename. This is followed by the
 * participant code, the guessing mode, the condition code, and the block code, for example,
 * <code>LetterGuessingExperiment-P01-One_guess-C01-S01-B01.sd2</code>.
 * <p>
 * 
 * <h3>Discussion</h3>
 * 
 * The data may be useful for follow-up analyses, for example, to estimate the user's skill in
 * guessing. One approach is to compare the user's guesses against those that could be programmed by
 * a language model. A simple language model might work with <a
 * href="d1-letterfreq.txt">letter-frequency</a>, <a href="d1-digramfreq.txt">digram-frequency</a>,
 * <a href="d1-trigramfreq.txt">trigram-frequency</a>, or <a
 * href="d1-quadgramfreq.txt">quadgram-frequency</a> lists. If guesses were done according to the
 * language model, would the result be better or worse than the user's result? To illustrate,
 * consider "picture" in the phrase "<i>a picture is worth a thousand words</i>". The
 * quadgram-frequency list includes eight non-zero quadgrams beginning with "pic". Sorted by
 * frequency, these are
 * 
 * <pre>
*     pict  14739
*     pica   9143
*     pick   6907
*     pic_   3598
*     pici   3498
*     pics   1838
*     picu    749
*     picn    567
* </pre>
 * 
 * The language model would work well here, since "t" is the first choice. Unfortunately, the
 * language model would not do as well on the next guess, since "u" appears in the third-ranked
 * quadgram beginning with "ict":
 * <p>
 * 
 * <pre>
*     icti  22098
*     ict_  15628
*     ictu  14739
*     icto  12354
*     icts   2828
*     icte   2746
*     ictl   1885
*     icta   1591
* </pre>
 * 
 * Furthermore, since humans inherently possess a vast knowledge of idioms and clichés in their
 * native language, the user is likely to fair quite well (better than the language model?) as
 * guessing proceeds deeper into the phrase. For example, most native speakers of English can easily
 * complete the phrase, "a picture is worth a thousand&nbsp;______". So, the comparison suggested
 * &mdash; between the language model and the user &mdash; could be pursued in an overall sense or
 * as a function of "position in phrase".
 * <p>
 * 
 * A more advanced language model might use a word list with part-of-speech (POS) tagging, as
 * described in <a href="http://www.yorku.ca/mack/nordichi2008.html"> Improved word list ordering
 * for text entry on ambiguous keyboards</a> (Gong, Tarasewich, and MacKenzie, <i>NordiCHI
 * 2008</i>).
 * <p>
 * 
 * The timestamp data might be useful in revealing how thoughtful the user was. Did the user think
 * carefully about the guesses, or did the user appear to be hurried and guessing randomly?
 * <p>
 * 
 * @author Scott MacKenzie, 2011-2015
 * @author Steven Castellucci, 2014
 */
public class LetterGuessingExperiment
{
	final static String APP_NAME = "LetterGuessingExperiment";
	final static int PARAMETERS = 6; // must equal the number of parameters defined below

	// setup parameters (1st value is default)
	final static String[] PARTICIPANT_CODES = { "P00", "P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09",
			"P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17", "P18", "P19", "P20", "P21", "P22", "P23", "P24",
			"P25", "P26", "P27", "P28", "P29", "P30" };
	final static String[] CONDITION_CODE = { "C00" };
	final static String[] NUMBER_OF_PHRASES = { "5" };
	final static String[] PHRASES_FILE = { "phrases2.txt" };
	final static String[] BEEP_ON_ERROR = { "true" };
	final static String[] GUESS_MODE = { "One guess", "Guess until correct" };

	// identify other files needed (used when execution from the .jar file for the 1st time)
	final static String[] OTHER_FILES = { "phrases2.txt", "d1-digramfreq.txt", "d1-letterfreq.txt",
			"d1-quadgramfreq.txt", "d1-trigramfreq.txt" };

	static final int ONE_GUESS = 100;
	static final int GUESS_UNTIL_CORRECT = 200;

	public static void main(String[] args) throws IOException
	{
		// use Win32 look and feel
		try
		{
			UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
		} catch (Exception e)
		{
		}

		/*
		 * Define the default parameters settings. These are only used if the .cfg file does not
		 * exit, such as when the application is launched for the 1st time.
		 */
		String[] defaultParameters = new String[PARAMETERS];
		defaultParameters[0] = PARTICIPANT_CODES[0];
		defaultParameters[1] = CONDITION_CODE[0];
		defaultParameters[2] = NUMBER_OF_PHRASES[0];
		defaultParameters[3] = PHRASES_FILE[0];
		defaultParameters[4] = BEEP_ON_ERROR[0];
		defaultParameters[5] = GUESS_MODE[0];

		Configuration c = Configuration.readConfigurationData(APP_NAME, defaultParameters, OTHER_FILES);
		if (c == null)
		{
			System.out.println("Error reading configuration data from " + APP_NAME + ".cfg!");
			System.exit(0);
		}

		// ----------------------------------------------------------
		// user the app's parameter settings to create a setup dialog
		// ----------------------------------------------------------

		SetupItemInfo[] sii = new SetupItemInfo[PARAMETERS];
		sii[0] = new SetupItemInfo(SetupItem.COMBO_BOX, "Participant code ", PARTICIPANT_CODES);
		sii[1] = new SetupItemInfo(SetupItem.TEXT_FIELD, "Condition code ", CONDITION_CODE);
		sii[2] = new SetupItemInfo(SetupItem.TEXT_FIELD, "Number of phrases", NUMBER_OF_PHRASES);
		sii[3] = new SetupItemInfo(SetupItem.TEXT_FIELD, "Phrases file ", PHRASES_FILE);
		sii[4] = new SetupItemInfo(SetupItem.CHECK_BOX, "Beep on error  ", BEEP_ON_ERROR);
		sii[5] = new SetupItemInfo(SetupItem.COMBO_BOX, "Guessing mode ", GUESS_MODE);

		// use setup to allow changes to the default or existing configuration
		Setup s = new Setup(null, c, APP_NAME, sii);
		s.showSetup(null);

		LetterGuessingExperimentGui screen = new LetterGuessingExperimentGui(c);
		screen.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		screen.setTitle(APP_NAME);
		screen.pack();

		// put application in center of screen
		int w = screen.getWidth();
		int h = screen.getHeight();
		Toolkit t = Toolkit.getDefaultToolkit();
		Dimension d = t.getScreenSize();
		screen.setLocation((d.width - w) / 2, (d.height - h) / 2);
		screen.setVisible(true);
	}
}

class LetterGuessingExperimentGui extends JFrame implements ActionListener
{
	private static final long serialVersionUID = 1L;

	final String SD1_HEADER = "";
	final String SD2_HEADER = "App,Participant,Mode,Condition,Block,Phrase,Text,Reduced_text,"
			+ "Correct_guesses,Incorrect_guesses,KSPC,Time\n";

	// configuration parameters
	String participantCode;
	String conditionCode;
	int numberOfPhrases;
	String phrasesFile;
	boolean beepOnError;
	int mode;
	String modeString;

	String blockCode;

	int phraseCount;
	String[] phraseArray;

	JTextField presentedTextField;
	JTextField reducedTextField;
	Random r = new Random();

	String[] phrases;
	BufferedWriter sd1File, sd2File;
	Configuration c;

	final Font TEXT_FIELD_FONT = new Font("monospaced", Font.PLAIN, 20);
	final int MAX_LENGTH = 50;

	String m = "";
	String textToGuess = ""; // written to output data file
	String presentedText = ""; // top field
	String reducedText = ""; // bottom field (and written to output data file)

	Clip errorSound;
	int letterCount;
	int guessCount;

	JDialog resultsDialog;
	JOptionPane resultsPane;
	JTextArea resultsArea;

	Keyboard k;
	String s1Result, s2Result, mode2Guesses;
	long elapsedTime, t;

	LetterGuessingExperimentGui(Configuration cArg) throws IOException
	{
		k = new Keyboard(50, 45, this);
		k.setAlignmentX(Component.CENTER_ALIGNMENT);

		letterCount = 0;
		phraseCount = 0;

		// initialize setup parameters
		c = cArg;
		participantCode = c.getConfigurationParameter(0);
		conditionCode = c.getConfigurationParameter(1);
		numberOfPhrases = Integer.parseInt(c.getConfigurationParameter(2));
		// NOTE: block code computed automatically
		phrasesFile = c.getConfigurationParameter(3);
		beepOnError = Boolean.valueOf(c.getConfigurationParameter(4));
		mode = -1;
		modeString = "";
		if (c.getConfigurationParameter(5).equals("One guess"))
		{
			mode = LetterGuessingExperiment.ONE_GUESS;
			modeString = "One_guess";
		} else
		{
			mode = LetterGuessingExperiment.GUESS_UNTIL_CORRECT;
			modeString = "Correct_guess";
		}

		// initialize sound clips
		errorSound = initSound("miss.wav");

		// open disk file for input
		BufferedReader inFile = new BufferedReader(new FileReader(phrasesFile));
		Vector<String> v = new Vector<String>();
		String s;
		while ((s = inFile.readLine()) != null)
			v.addElement(s);
		phrases = new String[v.size()];
		v.copyInto(phrases);

		// build an array of phrases for the experiment (no repeats!)
		phraseArray = new String[numberOfPhrases];
		for (int i = 0; i < phraseArray.length; ++i)
		{
			Random r = new Random();
			phraseArray[i] = phrases[r.nextInt(phrases.length)];
			for (int j = 0; j < i; ++j)
				while (phraseArray[i].equals(phraseArray[j]))
					phraseArray[i] = phrases[r.nextInt(phrases.length)];
		}

		// -------------------
		// open sd1, sd2 files
		// -------------------

		blockCode = "";
		int blockCodeNumber = 0;
		String base = LetterGuessingExperiment.APP_NAME + "-" + participantCode + "-" + modeString + "-"
				+ conditionCode;
		String s1 = "";
		String s2 = "";
		do // find next available block code
		{
			++blockCodeNumber;
			blockCode = blockCodeNumber < 10 ? "B0" + blockCodeNumber : "B" + blockCodeNumber;
			s1 = base + "-" + blockCode + ".sd1";
			s2 = base + "-" + blockCode + ".sd2";
		} while (new File(s1).exists());

		try
		{
			sd1File = new BufferedWriter(new FileWriter(s1));
			sd2File = new BufferedWriter(new FileWriter(s2));
		} catch (IOException e)
		{
			System.out.println("I/O error: can't open sd1/sd2 data files");
			System.exit(0);
		}

		try
		{
			sd1File.write(SD1_HEADER, 0, SD1_HEADER.length());
			sd1File.flush();
			sd2File.write(SD2_HEADER, 0, SD2_HEADER.length());
			sd2File.flush();
		} catch (IOException e)
		{
			System.err.println("ERROR WRITING TO DATA FILE!\n" + e);
			System.exit(1);
		}

		presentedTextField = new JTextField(MAX_LENGTH);
		presentedTextField.setFont(TEXT_FIELD_FONT);
		presentedTextField.setEditable(false);
		presentedTextField.setForeground(new Color(0, 0, 128));
		presentedTextField.setBackground(Color.WHITE);

		reducedTextField = new JTextField(MAX_LENGTH);
		reducedTextField.setFont(TEXT_FIELD_FONT);
		reducedTextField.setEditable(true); // allows I-beam (caret) to appear
		reducedTextField.setForeground(new Color(0, 0, 128));
		reducedTextField.setBackground(Color.WHITE);

		resultsArea = new JTextArea(9, 20);
		resultsArea.setFont(new Font("sansserif", Font.PLAIN, 18));
		resultsArea.setBackground((new JButton()).getBackground());
		resultsPane = new JOptionPane(resultsArea, JOptionPane.INFORMATION_MESSAGE);
		resultsPane.setFont(new Font("sansserif", Font.PLAIN, 18));
		resultsDialog = resultsPane.createDialog(this, "Information");

		// prepare for first trial
		newTrial();

		JPanel p2 = new JPanel();
		p2.setLayout(new BoxLayout(p2, BoxLayout.Y_AXIS));
		p2.add(Box.createRigidArea(new Dimension(0, 10)));
		p2.add(presentedTextField);
		p2.add(Box.createRigidArea(new Dimension(0, 10)));
		p2.add(reducedTextField);
		p2.add(Box.createRigidArea(new Dimension(0, 10)));

		JPanel p3 = new JPanel();
		p3.add(k);

		JPanel p4 = new JPanel(new BorderLayout());
		p4.add("North", p2);
		p4.add("South", p3);
		p4.add(Box.createRigidArea(new Dimension(0, 10)));
		p4.setBorder(BorderFactory.createEmptyBorder(30, 30, 30, 30));
		this.setContentPane(p4);
	}

	void showError(String msg)
	{
		JOptionPane.showMessageDialog(null, msg, "I/O Error", JOptionPane.ERROR_MESSAGE);
	}

	// -------------------------------
	// implement ActionListener method
	// -------------------------------

	public void actionPerformed(ActionEvent ae)
	{
		// get character for soft keypress
		JButton jb = (JButton)ae.getSource();
		String s = jb.getText();
		char c = (s.toLowerCase()).charAt(0);
		long now = System.currentTimeMillis();
		elapsedTime += now - t; // elapsed time
		t = now;

		// don't do anything if past the end of the text (just in case)
		if (letterCount == textToGuess.length())
			return;

		char c1 = textToGuess.charAt(letterCount);
		if (c1 == ' ')
			c1 = '_'; // show spaces as underscores
		char c2 = c == ' ' ? '_' : c;

		if (mode == LetterGuessingExperiment.ONE_GUESS)
		{
			s1Result += (c1 + "," + c2 + "," + elapsedTime) + "\n";
			if (c != textToGuess.charAt(letterCount)) // incorrect guess
			{
				if (beepOnError)
					playSound(errorSound);
				reducedText += "" + textToGuess.charAt(letterCount);
			} else
			// correct guess
			{
				reducedText += "-";
			}
			++letterCount;

			presentedText = textToGuess.substring(0, letterCount) + presentedText.substring(letterCount);
			presentedTextField.setText(presentedText);
			reducedTextField.setText(reducedText);
			reducedTextField.requestFocus();
		} else
		// Mode = CORRECT_GUESS (Guess until correct)
		{
			mode2Guesses += "" + c2;
			++guessCount;
			if (guessCount == 1)
				reducedText += "" + guessCount;
			else
			{
				reducedText = reducedText.substring(0, reducedText.length() - 1);
				reducedText += guessCount < 10 ? guessCount : "" + (char)((guessCount - 10) + 'A');
			}
			if (c != textToGuess.charAt(letterCount)) // incorrect guess
			{
				k.disableKey(jb, this);
				if (beepOnError)
					playSound(errorSound);
			} else
			// correct guess
			{
				s1Result += (c1 + "," + mode2Guesses + "," + elapsedTime) + "\n";
				mode2Guesses = "";
				++letterCount;
				guessCount = 0;
				k.enableKeys(this);
			}

			presentedText = textToGuess.substring(0, letterCount) + presentedText.substring(letterCount);
			presentedTextField.setText(presentedText);
			reducedTextField.setText(reducedText);
			reducedTextField.requestFocus();
		}

		if (letterCount == presentedText.length()) // end of text phrase
		{
			// tally correct/incorrect/kspc stats
			int correct = 0;
			int incorrect = 0;
			double kspc = 0.0;
			if (mode == LetterGuessingExperiment.ONE_GUESS)
			{
				for (int i = 0; i < reducedText.length(); ++i)
					if (reducedText.charAt(i) == '-')
						++correct;
					else
						++incorrect;
				kspc = 1.0; // always 1 for One-guess mode
			} else
			// mode = CORRECT_GUESS
			{
				correct = reducedText.length();
				for (int i = 0; i < reducedText.length(); ++i)
				{
					char tmp = reducedText.charAt(i);
					int n;
					if (tmp >= '0' && tmp <= '9')
						n = (tmp - '0');
					else
						n = (tmp - 'A') + 10;
					incorrect += (n - 1);
					kspc += (n - 1) + 1; // +1 for final correct guess
				}
				kspc /= reducedText.length();
			}

			String modeString = mode == LetterGuessingExperiment.ONE_GUESS ? "One_guess" : "Correct_guess";

			s2Result = LetterGuessingExperiment.APP_NAME + "," + participantCode + "," + modeString + ","
					+ conditionCode + "," + blockCode + "," + (phraseCount + 1) + "," + textToGuess + "," + reducedText
					+ "," + correct + "," + incorrect + "," + kspc + "," + elapsedTime + "\n";

			// write to data files
			try
			{
				sd1File.write(s1Result, 0, s1Result.length());
				sd1File.flush();
				sd2File.write(s2Result, 0, s2Result.length());
				sd2File.flush();
			} catch (IOException e)
			{
				showError("ERROR WRITING TO DATA FILE!\n" + e);
				System.exit(1);
			}

			s = "Thank you!\n\n";
			s += String.format("Number of letters:\t%d\n", reducedText.length());
			s += String.format("Correct guesses:\t%d\n", correct);
			s += String.format("Incorrect guesses:\t%d\n", incorrect);
			s += String.format("KSPC:\t\t%.2f\n\n", kspc);
			s += "Click OK to continue";
			resultsArea.setText(s);

			resultsDialog.setVisible(true);

			++phraseCount;
			if (phraseCount == numberOfPhrases)
			{
				try
				{
					sd1File.close();
				} catch (IOException e)
				{
					showError("ERROR CLOSING DATA FILE!\n" + e);
					System.exit(1);
				}
				System.exit(0);
			} else
				newTrial();
		}
	}

	// Return the entropy of the reduced text in bits per letter.
	// This is just a rough first crack at the problem. Any ideas?
	double entropy(String reducedTextArg, int mode)
	{
		String s = reducedTextArg;
		double entropy = 0.0;
		final double LOG2_27 = Math.log(27.0) / Math.log(2);

		for (int i = 0; i < s.length(); ++i)
		{
			char c = s.charAt(i);
			if (mode == LetterGuessingExperiment.ONE_GUESS)
			{
				entropy += c == '-' ? 0.0 : LOG2_27;
			} else
			// MULTI_GUESS mode
			{
				int n = c <= '9' ? (c - '0') : (c - 'A') + 10;
				n -= 1; // adjust so 0 is 1st guess correct, 1 is 2nd guess correct, etc.
				entropy += n == 0 ? 0.0 : Math.log(2.0 * n) / Math.log(2.0);
			}
		}
		entropy /= s.length();
		return entropy;
	}

	void newTrial()
	{
		textToGuess = phraseArray[phraseCount].toLowerCase();
		presentedText = "";
		for (int i = 0; i < textToGuess.length(); ++i)
			presentedText += "*";
		reducedText = "";
		letterCount = 0;
		guessCount = 0;
		presentedTextField.setText(presentedText);
		reducedTextField.setText("");
		reducedTextField.requestFocus();
		s1Result = textToGuess + "\n";
		s2Result = "";
		mode2Guesses = "";
		t = System.currentTimeMillis();
		elapsedTime = 0;
	}

	// Initialize stream for sound clip
	public Clip initSound(String soundFile)
	{
		AudioInputStream audioInputStream;
		Clip c = null;
		try
		{
			// Added for executable Jar:
			audioInputStream = AudioSystem.getAudioInputStream(new BufferedInputStream(getClass().getResourceAsStream(
					soundFile)));
			c = AudioSystem.getClip();
			c.open(audioInputStream);
		} catch (Exception e)
		{
			showError("ERROR: Unable to load sound clip (" + soundFile + ")");
		}
		return c;
	}

	// play sound
	public void playSound(Clip c)
	{
		if (c != null)
		{
			c.setFramePosition(0); // rewind clip to beginning
			c.start(); // stops at end of clip
		}
	}

	class Keyboard extends JPanel implements ActionListener
	{
		private static final long serialVersionUID = 1L;

		final String keyLabel = "QWERTYUIOPASDFGHJKLZXCVBNM";
		JButton[] key;
		JButton spaceKey;
		int width, height;

		// -----------
		// constructor
		// -----------

		public Keyboard(int widthArg, int heightArg, ActionListener alArg)
		{
			width = widthArg;
			height = heightArg;

			key = new JButton[keyLabel.length()];
			for (int i = 0; i < key.length; ++i)
			{
				key[i] = new JButton(keyLabel.substring(i, i + 1));
				key[i].setBackground(Color.LIGHT_GRAY);
				key[i].setFont(new Font("sansserif", Font.BOLD, 16));
				key[i].setPreferredSize(new Dimension(width, height));
			}
			spaceKey = new JButton(" ");
			spaceKey.setBackground(Color.LIGHT_GRAY);
			spaceKey.setPreferredSize(new Dimension(4 * width, height));

			// install listeners (from instantiating class; see constructor args)
			for (int i = 0; i < key.length; ++i)
				key[i].addActionListener(alArg);
			spaceKey.addActionListener(alArg);

			this.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
			this.setLayout(null);
			int x;
			int y;
			for (int i = 0; i < key.length; ++i)
			{
				if (i < 10)
				{
					x = i * width;
					y = 0;
				} else if (i < 19)
				{
					x = (i - 10) * width + (width / 3);
					y = height;
				} else
				{
					x = (i - 19) * width + 2 * (width / 3);
					y = 2 * height;
				}
				this.add(key[i]);
				key[i].setBounds(x, y, width, height);
			}
			this.add(spaceKey);
			spaceKey.setBounds(2 * width, 3 * height, 6 * width, height);
			this.setPreferredSize(new Dimension(10 * width, 4 * height));
			this.setMaximumSize(new Dimension(10 * width, 4 * height));
			this.setBorder(BorderFactory.createLineBorder(Color.gray));
		}

		ActionEvent ae;

		public ActionEvent getActionEvent()
		{
			return ae;
		}

		public void setActionEvent(ActionEvent aeArg)
		{
			ae = aeArg;
		}

		public void disableKey(JButton keyArg, ActionListener alArg)
		{
			keyArg.setBackground(Color.DARK_GRAY);
			keyArg.setOpaque(true); // needed for Mac OS
			// keyArg.setBorderPainted(false);
		}

		public void enableKeys(ActionListener alArg)
		{
			for (int i = 0; i < key.length; ++i)
				key[i].setBackground(Color.LIGHT_GRAY);
			spaceKey.setBackground(Color.LIGHT_GRAY);
		}

		// -------------------------------
		// implement ActionListener method
		// -------------------------------

		public void actionPerformed(ActionEvent ae)
		{
			setActionEvent(ae);
		}
	}
}