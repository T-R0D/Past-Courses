using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Data.SqlClient;


namespace HelloDatabase
{

    public partial class MainForm : Form
    {
        // this only works when we are on campus ans should work for any user with access
        // rights to the particular database. This will use the username/password of the
        // user logged in to the Windows account using this application.
        private  String databaseName;
        private  String CONNECTION_STRING;
        SqlConnection dbConnection = null;

        public MainForm()
        {
            InitializeComponent();
            radioButton_thenriod.Checked = true;
            databaseName = "thenriod";
            dbConnection = new SqlConnection();
        }

        private void GetIssueReportButton_Click(object sender, EventArgs e)
        {
            int issueReportId;
            bool parseSuccess = Int32.TryParse(IssueReportIdInput.Text, out issueReportId);

            if (parseSuccess)
            {
                CONNECTION_STRING = String.Format(@"Data Source=ISSQL\Students;Initial Catalog={0};Integrated Security=SSPI", databaseName);

                using (SqlConnection connection = new SqlConnection(CONNECTION_STRING))
                using (SqlCommand command = new SqlCommand("stpr_FindIssueReport", connection))
                {
                    connection.Open();

                    if (connection.State == ConnectionState.Open)
                    {
                        command.CommandType = CommandType.StoredProcedure;
                        command.Parameters.AddWithValue("@ReportId", issueReportId);

                        SqlDataReader reader = command.ExecuteReader();

                        String result = String.Empty;
                        if (reader.HasRows)
                        {
                            reader.Read();
                            for (int i = 0; i < reader.FieldCount; ++i)
                            {
                                result += reader.GetName(i) + ": " + reader[i].ToString() + Environment.NewLine;
                            }
                        }
                        else
                        {
                            result = "The Id provided is invalid.";
                        }
                        OutputDumpBox.Text = result;

                    }
                    connection.Close();
                }
            }
            else
            {
                OutputDumpBox.Text = "Please enter a number for the issue report Id.";
            }
        }

        private void TestFormCompleteButton_Click(object sender, EventArgs e)
        {
            String outcome = TestResultsBox.Text;
            String recommendation = RecommendedResolutionBox.Text;
            int testId;
            bool parseSuccess = Int32.TryParse(TestFormIdBox.Text, out testId);

            if (parseSuccess)
            {
                CONNECTION_STRING = String.Format(@"Data Source=ISSQL\Students;Initial Catalog={0};Integrated Security=SSPI", databaseName);

                using (SqlConnection connection = new SqlConnection(CONNECTION_STRING))
                using (SqlCommand command = new SqlCommand("stpr_MarkTestAsComplete", connection))
                {
                    connection.Open();

                    if (connection.State == ConnectionState.Open)
                    {
                        command.CommandType = CommandType.StoredProcedure;
                        command.Parameters.AddWithValue("@TestFormId", testId);
                        command.Parameters.AddWithValue("@TestOutcome", outcome);
                        command.Parameters.AddWithValue("@Recommendation", recommendation);

                        int numRowsUpdated = command.ExecuteNonQuery();

                        if (numRowsUpdated == 1)
                        {
                            OutputDumpBox.Text = "Job marked as complete.";
                        }
                        else
                        {
                            OutputDumpBox.Text = "Unable to mark that test as complete. " +
                                "Are you sure it exists or hasn't already been completed?";
                        }
                    }
                    connection.Close();
                }
            }
            else
            {
                OutputDumpBox.Text = "Please enter a number for the Test Form Id.";
            }

        }

        private void button_insert_unit_Click(object sender, EventArgs e)
        {

            if (textBox_serialNumber.Text != String.Empty && textBox_modelNumber.Text != String.Empty)
            {
                String insertCommand = "INSERT INTO a_unit VALUES(@SerialNumber, @ModelNumber,@Date)";
                CONNECTION_STRING = String.Format(@"Data Source=ISSQL\Students;Initial Catalog={0};Integrated Security=SSPI", databaseName);

                using (SqlConnection connection = new SqlConnection(CONNECTION_STRING))
                using (SqlCommand command = new SqlCommand(insertCommand, connection))
                {
                    connection.Open();

                    if (connection.State == ConnectionState.Open)
                    {
                        try
                        {
                            command.Parameters.AddWithValue("@SerialNumber", textBox_serialNumber.Text);
                            command.Parameters.AddWithValue("@ModelNumber", textBox_modelNumber.Text);
                            command.Parameters.AddWithValue("@Date", DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));
                            command.ExecuteNonQuery();
                        } 
                        catch(SqlException exception)
                        {
                            MessageBox.Show(exception.Message);
                        }
                    }
                    connection.Close();
                }
            }
            else
            {
                MessageBox.Show("Please Enter a Valid Serial Number or Model Number");
            }

        }

        private void button_unit_log_Click(object sender, EventArgs e)
        {
             String insertCommand = "SELECT * FROM a_UnitLog";
             CONNECTION_STRING = String.Format(@"Data Source=ISSQL\Students;Initial Catalog={0};Integrated Security=SSPI", databaseName);

                using (SqlConnection connection = new SqlConnection(CONNECTION_STRING))
                using (SqlCommand command = new SqlCommand(insertCommand, connection))
                {
                    connection.Open();

                    String result = String.Empty;

                    if (connection.State == ConnectionState.Open)
                    {

                        SqlDataReader reader = command.ExecuteReader();
                      
                        while (reader.Read())
                        {
                            for(int i = 0; i < reader.FieldCount; i++)
                            {
                                result += String.Format("{0}: {1} {2}",reader.GetName(i),reader[i],Environment.NewLine);
                            }
                            result += "---------------------------------------------------------------------------------" + Environment.NewLine;
                        }

                        OutputDumpBox.Text = result;
                    }
                    connection.Close();
                }       
        }

        private void radioButton_thenriod_CheckedChanged(object sender, EventArgs e)
        {
            databaseName = "thenriod";
        }

        private void radioButton_jsantoyo_CheckedChanged(object sender, EventArgs e)
        {
            databaseName = "jsantoyo";
        }

        private void radioButton_rajas_CheckedChanged(object sender, EventArgs e)
        {
            databaseName = "rajas";
        }
    }
}
