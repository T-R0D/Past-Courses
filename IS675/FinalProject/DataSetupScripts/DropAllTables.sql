-- Drop Tables
DROP TABLE a_UnitLog;
DROP TABLE Employee_a_TestForm_Audit;
DROP TABLE a_TestForm;
DROP TABLE a_IssueReport;
DROP TABLE a_ReportMethod;
DROP TABLE a_IssueReportProblem;
DROP TABLE a_ReturnReasonSelection;
DROP TABLE a_Unit;
DROP TABLE a_ReasonReturned;
DROP TABLE a_Model;
DROP TABLE a_Distributor;
DROP TABLE a_Employee;
DROP TABLE a_Customer;
DROP TABLE a_Location;


--Drop Views
DROP VIEW av_ReturnedUnits;
DROP VIEW av_TestCounts;
DROP VIEW av_Number3;
DROP VIEW av_Number5;
DROP VIEW v_ModelReportCount;
DROP VIEW v_ModelTestCount;
DROP VIEW v_ModelInjuryCount;
DROP VIEW v_ModelReportDate;
DROP VIEW v_ModelSummary;
DROP VIEW v_ModelMinReport;
DROP VIEW v_ModelMaxReport;
DROP VIEW av_ReporterTypeFrequency;

-- Drop Triggers
DROP TRIGGER unit_insert_log;
drop trigger trgAfterUpdate;

--Drop Procedures

DROP PROCEDURE stpr_MarkTestAsComplete;
DROP PROCEDURE stpr_oldest_open_report;
DROP PROCEDURE stpr_IncompleteReports;
DROP PROCEDURE stpr_FindIssueReport;
DROP PROCEDURE stpr_InsertIntoIssueReport;