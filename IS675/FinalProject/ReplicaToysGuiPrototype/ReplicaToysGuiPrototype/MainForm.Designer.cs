namespace HelloDatabase
{
    partial class MainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.OutputDumpBox = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.IssueReportIdInput = new System.Windows.Forms.TextBox();
            this.GetIssueReportButton = new System.Windows.Forms.Button();
            this.GetIssueReportCue = new System.Windows.Forms.Label();
            this.Stpr1Group = new System.Windows.Forms.GroupBox();
            this.OutputGroup = new System.Windows.Forms.GroupBox();
            this.InsertUpdateTestGroup = new System.Windows.Forms.GroupBox();
            this.TestFormCompleteButton = new System.Windows.Forms.Button();
            this.RecommendedResolutionBox = new System.Windows.Forms.TextBox();
            this.TestResultsBox = new System.Windows.Forms.TextBox();
            this.TestFormIdBox = new System.Windows.Forms.TextBox();
            this.MadeByToolTip = new System.Windows.Forms.ToolTip(this.components);
            this.InsertUnitGroup = new System.Windows.Forms.GroupBox();
            this.button_unit_log = new System.Windows.Forms.Button();
            this.button_insert_unit = new System.Windows.Forms.Button();
            this.textBox_modelNumber = new System.Windows.Forms.TextBox();
            this.ModelLabel = new System.Windows.Forms.Label();
            this.textBox_serialNumber = new System.Windows.Forms.TextBox();
            this.SerialNumberlabel = new System.Windows.Forms.Label();
            this.groupBox_Database = new System.Windows.Forms.GroupBox();
            this.radioButton_rajas = new System.Windows.Forms.RadioButton();
            this.radioButton_jsantoyo = new System.Windows.Forms.RadioButton();
            this.radioButton_thenriod = new System.Windows.Forms.RadioButton();
            this.Stpr1Group.SuspendLayout();
            this.OutputGroup.SuspendLayout();
            this.InsertUpdateTestGroup.SuspendLayout();
            this.InsertUnitGroup.SuspendLayout();
            this.groupBox_Database.SuspendLayout();
            this.SuspendLayout();
            // 
            // OutputDumpBox
            // 
            this.OutputDumpBox.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.OutputDumpBox.Location = new System.Drawing.Point(9, 44);
            this.OutputDumpBox.Multiline = true;
            this.OutputDumpBox.Name = "OutputDumpBox";
            this.OutputDumpBox.ReadOnly = true;
            this.OutputDumpBox.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.OutputDumpBox.Size = new System.Drawing.Size(823, 119);
            this.OutputDumpBox.TabIndex = 1;
            this.OutputDumpBox.Tag = "";
            this.MadeByToolTip.SetToolTip(this.OutputDumpBox, "Lovingly made by Terence Henriod and Jorge Santoyo. And Raja I guess.");
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.BackColor = System.Drawing.SystemColors.WindowFrame;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 15.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.ForeColor = System.Drawing.SystemColors.ButtonFace;
            this.label1.Location = new System.Drawing.Point(9, 16);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(296, 25);
            this.label1.TabIndex = 3;
            this.label1.Text = "See (any and all) output here:";
            this.MadeByToolTip.SetToolTip(this.label1, "Lovingly made by Terence Henriod and Jorge Santoyo. And Raja I guess.");
            // 
            // IssueReportIdInput
            // 
            this.IssueReportIdInput.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.IssueReportIdInput.Location = new System.Drawing.Point(6, 34);
            this.IssueReportIdInput.Name = "IssueReportIdInput";
            this.IssueReportIdInput.Size = new System.Drawing.Size(192, 22);
            this.IssueReportIdInput.TabIndex = 4;
            // 
            // GetIssueReportButton
            // 
            this.GetIssueReportButton.Location = new System.Drawing.Point(242, 54);
            this.GetIssueReportButton.Name = "GetIssueReportButton";
            this.GetIssueReportButton.Size = new System.Drawing.Size(125, 23);
            this.GetIssueReportButton.TabIndex = 5;
            this.GetIssueReportButton.Text = "Get an Issue Report";
            this.GetIssueReportButton.UseVisualStyleBackColor = true;
            this.GetIssueReportButton.Click += new System.EventHandler(this.GetIssueReportButton_Click);
            // 
            // GetIssueReportCue
            // 
            this.GetIssueReportCue.AutoSize = true;
            this.GetIssueReportCue.BackColor = System.Drawing.SystemColors.InactiveCaptionText;
            this.GetIssueReportCue.Font = new System.Drawing.Font("Microsoft Sans Serif", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.GetIssueReportCue.ForeColor = System.Drawing.SystemColors.ButtonFace;
            this.GetIssueReportCue.Location = new System.Drawing.Point(6, 15);
            this.GetIssueReportCue.Name = "GetIssueReportCue";
            this.GetIssueReportCue.Size = new System.Drawing.Size(192, 16);
            this.GetIssueReportCue.TabIndex = 6;
            this.GetIssueReportCue.Text = "Enter an Issue Report Id (like 5)";
            // 
            // Stpr1Group
            // 
            this.Stpr1Group.Controls.Add(this.GetIssueReportCue);
            this.Stpr1Group.Controls.Add(this.IssueReportIdInput);
            this.Stpr1Group.Controls.Add(this.GetIssueReportButton);
            this.Stpr1Group.Location = new System.Drawing.Point(17, 12);
            this.Stpr1Group.Name = "Stpr1Group";
            this.Stpr1Group.Size = new System.Drawing.Size(382, 83);
            this.Stpr1Group.TabIndex = 7;
            this.Stpr1Group.TabStop = false;
            this.Stpr1Group.Text = "Stored Procedure 1";
            // 
            // OutputGroup
            // 
            this.OutputGroup.Controls.Add(this.label1);
            this.OutputGroup.Controls.Add(this.OutputDumpBox);
            this.OutputGroup.Location = new System.Drawing.Point(12, 334);
            this.OutputGroup.Name = "OutputGroup";
            this.OutputGroup.Size = new System.Drawing.Size(832, 239);
            this.OutputGroup.TabIndex = 9;
            this.OutputGroup.TabStop = false;
            this.MadeByToolTip.SetToolTip(this.OutputGroup, "Lovingly made by Terence Henriod and Jorge Santoyo. And Raja I guess.");
            // 
            // InsertUpdateTestGroup
            // 
            this.InsertUpdateTestGroup.Controls.Add(this.TestFormCompleteButton);
            this.InsertUpdateTestGroup.Controls.Add(this.RecommendedResolutionBox);
            this.InsertUpdateTestGroup.Controls.Add(this.TestResultsBox);
            this.InsertUpdateTestGroup.Controls.Add(this.TestFormIdBox);
            this.InsertUpdateTestGroup.Location = new System.Drawing.Point(17, 101);
            this.InsertUpdateTestGroup.Name = "InsertUpdateTestGroup";
            this.InsertUpdateTestGroup.Size = new System.Drawing.Size(382, 197);
            this.InsertUpdateTestGroup.TabIndex = 10;
            this.InsertUpdateTestGroup.TabStop = false;
            this.InsertUpdateTestGroup.Text = "Stored Procedure 3";
            // 
            // TestFormCompleteButton
            // 
            this.TestFormCompleteButton.Location = new System.Drawing.Point(219, 67);
            this.TestFormCompleteButton.Name = "TestFormCompleteButton";
            this.TestFormCompleteButton.Size = new System.Drawing.Size(92, 61);
            this.TestFormCompleteButton.TabIndex = 4;
            this.TestFormCompleteButton.Text = "Mark Test as Complete";
            this.TestFormCompleteButton.UseVisualStyleBackColor = true;
            this.TestFormCompleteButton.Click += new System.EventHandler(this.TestFormCompleteButton_Click);
            // 
            // RecommendedResolutionBox
            // 
            this.RecommendedResolutionBox.Location = new System.Drawing.Point(6, 113);
            this.RecommendedResolutionBox.Multiline = true;
            this.RecommendedResolutionBox.Name = "RecommendedResolutionBox";
            this.RecommendedResolutionBox.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.RecommendedResolutionBox.Size = new System.Drawing.Size(165, 60);
            this.RecommendedResolutionBox.TabIndex = 3;
            this.RecommendedResolutionBox.Text = "Recommended problem resolution (anything is fine)";
            // 
            // TestResultsBox
            // 
            this.TestResultsBox.Location = new System.Drawing.Point(6, 54);
            this.TestResultsBox.Multiline = true;
            this.TestResultsBox.Name = "TestResultsBox";
            this.TestResultsBox.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.TestResultsBox.Size = new System.Drawing.Size(159, 53);
            this.TestResultsBox.TabIndex = 2;
            this.TestResultsBox.Text = "Test outcome description (anything is fine)";
            // 
            // TestFormIdBox
            // 
            this.TestFormIdBox.Location = new System.Drawing.Point(6, 28);
            this.TestFormIdBox.Name = "TestFormIdBox";
            this.TestFormIdBox.Size = new System.Drawing.Size(100, 20);
            this.TestFormIdBox.TabIndex = 0;
            this.TestFormIdBox.Text = "Test Form Id (try 5)";
            // 
            // InsertUnitGroup
            // 
            this.InsertUnitGroup.Controls.Add(this.button_unit_log);
            this.InsertUnitGroup.Controls.Add(this.button_insert_unit);
            this.InsertUnitGroup.Controls.Add(this.textBox_modelNumber);
            this.InsertUnitGroup.Controls.Add(this.ModelLabel);
            this.InsertUnitGroup.Controls.Add(this.textBox_serialNumber);
            this.InsertUnitGroup.Controls.Add(this.SerialNumberlabel);
            this.InsertUnitGroup.Location = new System.Drawing.Point(449, 101);
            this.InsertUnitGroup.Name = "InsertUnitGroup";
            this.InsertUnitGroup.Size = new System.Drawing.Size(395, 197);
            this.InsertUnitGroup.TabIndex = 11;
            this.InsertUnitGroup.TabStop = false;
            this.InsertUnitGroup.Text = "Insert Unit";
            // 
            // button_unit_log
            // 
            this.button_unit_log.Location = new System.Drawing.Point(10, 129);
            this.button_unit_log.Name = "button_unit_log";
            this.button_unit_log.Size = new System.Drawing.Size(104, 23);
            this.button_unit_log.TabIndex = 5;
            this.button_unit_log.Text = "View Insert Log";
            this.button_unit_log.UseVisualStyleBackColor = true;
            this.button_unit_log.Click += new System.EventHandler(this.button_unit_log_Click);
            // 
            // button_insert_unit
            // 
            this.button_insert_unit.Location = new System.Drawing.Point(143, 129);
            this.button_insert_unit.Name = "button_insert_unit";
            this.button_insert_unit.Size = new System.Drawing.Size(111, 23);
            this.button_insert_unit.TabIndex = 4;
            this.button_insert_unit.Text = "Insert!";
            this.button_insert_unit.UseVisualStyleBackColor = true;
            this.button_insert_unit.Click += new System.EventHandler(this.button_insert_unit_Click);
            // 
            // textBox_modelNumber
            // 
            this.textBox_modelNumber.Location = new System.Drawing.Point(125, 67);
            this.textBox_modelNumber.Name = "textBox_modelNumber";
            this.textBox_modelNumber.Size = new System.Drawing.Size(129, 20);
            this.textBox_modelNumber.TabIndex = 3;
            // 
            // ModelLabel
            // 
            this.ModelLabel.AutoSize = true;
            this.ModelLabel.Location = new System.Drawing.Point(6, 67);
            this.ModelLabel.Name = "ModelLabel";
            this.ModelLabel.Size = new System.Drawing.Size(79, 13);
            this.ModelLabel.TabIndex = 2;
            this.ModelLabel.Text = "Model Number:";
            // 
            // textBox_serialNumber
            // 
            this.textBox_serialNumber.Location = new System.Drawing.Point(125, 20);
            this.textBox_serialNumber.Name = "textBox_serialNumber";
            this.textBox_serialNumber.Size = new System.Drawing.Size(129, 20);
            this.textBox_serialNumber.TabIndex = 1;
            // 
            // SerialNumberlabel
            // 
            this.SerialNumberlabel.AutoSize = true;
            this.SerialNumberlabel.Location = new System.Drawing.Point(7, 20);
            this.SerialNumberlabel.Name = "SerialNumberlabel";
            this.SerialNumberlabel.Size = new System.Drawing.Size(76, 13);
            this.SerialNumberlabel.TabIndex = 0;
            this.SerialNumberlabel.Text = "Serial Number:";
            // 
            // groupBox_Database
            // 
            this.groupBox_Database.Controls.Add(this.radioButton_rajas);
            this.groupBox_Database.Controls.Add(this.radioButton_jsantoyo);
            this.groupBox_Database.Controls.Add(this.radioButton_thenriod);
            this.groupBox_Database.Location = new System.Drawing.Point(449, 12);
            this.groupBox_Database.Name = "groupBox_Database";
            this.groupBox_Database.Size = new System.Drawing.Size(395, 83);
            this.groupBox_Database.TabIndex = 12;
            this.groupBox_Database.TabStop = false;
            this.groupBox_Database.Text = "Select Database";
            // 
            // radioButton_rajas
            // 
            this.radioButton_rajas.AutoSize = true;
            this.radioButton_rajas.Location = new System.Drawing.Point(314, 34);
            this.radioButton_rajas.Name = "radioButton_rajas";
            this.radioButton_rajas.Size = new System.Drawing.Size(47, 17);
            this.radioButton_rajas.TabIndex = 2;
            this.radioButton_rajas.TabStop = true;
            this.radioButton_rajas.Text = "rajas";
            this.radioButton_rajas.UseVisualStyleBackColor = true;
            this.radioButton_rajas.CheckedChanged += new System.EventHandler(this.radioButton_rajas_CheckedChanged);
            // 
            // radioButton_jsantoyo
            // 
            this.radioButton_jsantoyo.AutoSize = true;
            this.radioButton_jsantoyo.Location = new System.Drawing.Point(165, 34);
            this.radioButton_jsantoyo.Name = "radioButton_jsantoyo";
            this.radioButton_jsantoyo.Size = new System.Drawing.Size(64, 17);
            this.radioButton_jsantoyo.TabIndex = 1;
            this.radioButton_jsantoyo.TabStop = true;
            this.radioButton_jsantoyo.Text = "jsantoyo";
            this.radioButton_jsantoyo.UseVisualStyleBackColor = true;
            this.radioButton_jsantoyo.CheckedChanged += new System.EventHandler(this.radioButton_jsantoyo_CheckedChanged);
            // 
            // radioButton_thenriod
            // 
            this.radioButton_thenriod.AutoSize = true;
            this.radioButton_thenriod.Location = new System.Drawing.Point(40, 34);
            this.radioButton_thenriod.Name = "radioButton_thenriod";
            this.radioButton_thenriod.Size = new System.Drawing.Size(63, 17);
            this.radioButton_thenriod.TabIndex = 0;
            this.radioButton_thenriod.TabStop = true;
            this.radioButton_thenriod.Text = "thenriod";
            this.radioButton_thenriod.UseVisualStyleBackColor = true;
            this.radioButton_thenriod.CheckedChanged += new System.EventHandler(this.radioButton_thenriod_CheckedChanged);
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(859, 580);
            this.Controls.Add(this.groupBox_Database);
            this.Controls.Add(this.InsertUnitGroup);
            this.Controls.Add(this.InsertUpdateTestGroup);
            this.Controls.Add(this.OutputGroup);
            this.Controls.Add(this.Stpr1Group);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "MainForm";
            this.Text = "Team08\'s Awesome Database Application";
            this.Stpr1Group.ResumeLayout(false);
            this.Stpr1Group.PerformLayout();
            this.OutputGroup.ResumeLayout(false);
            this.OutputGroup.PerformLayout();
            this.InsertUpdateTestGroup.ResumeLayout(false);
            this.InsertUpdateTestGroup.PerformLayout();
            this.InsertUnitGroup.ResumeLayout(false);
            this.InsertUnitGroup.PerformLayout();
            this.groupBox_Database.ResumeLayout(false);
            this.groupBox_Database.PerformLayout();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.TextBox OutputDumpBox;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox IssueReportIdInput;
        private System.Windows.Forms.Button GetIssueReportButton;
        private System.Windows.Forms.Label GetIssueReportCue;
        private System.Windows.Forms.GroupBox Stpr1Group;
        private System.Windows.Forms.GroupBox OutputGroup;
        private System.Windows.Forms.GroupBox InsertUpdateTestGroup;
        private System.Windows.Forms.Button TestFormCompleteButton;
        private System.Windows.Forms.TextBox RecommendedResolutionBox;
        private System.Windows.Forms.TextBox TestResultsBox;
        private System.Windows.Forms.TextBox TestFormIdBox;
        private System.Windows.Forms.ToolTip MadeByToolTip;
        private System.Windows.Forms.GroupBox InsertUnitGroup;
        private System.Windows.Forms.Button button_insert_unit;
        private System.Windows.Forms.TextBox textBox_modelNumber;
        private System.Windows.Forms.Label ModelLabel;
        private System.Windows.Forms.TextBox textBox_serialNumber;
        private System.Windows.Forms.Label SerialNumberlabel;
        private System.Windows.Forms.Button button_unit_log;
        private System.Windows.Forms.GroupBox groupBox_Database;
        private System.Windows.Forms.RadioButton radioButton_rajas;
        private System.Windows.Forms.RadioButton radioButton_jsantoyo;
        private System.Windows.Forms.RadioButton radioButton_thenriod;
    }
}

