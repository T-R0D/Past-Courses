CREATE TABLE a_Location(
	ZipCode int NOT NULL,
	City varchar(100) NOT NULL,
	[State] varchar(2) NOT NULL
	CONSTRAINT pk_ZipCode PRIMARY KEY(ZipCode)
);

CREATE TABLE a_Customer(
	CustomerId int NOT NULL,
	ZipCode int NOT NULL,
	FirstName varchar(60) NOT NULL,
	LastName varchar(60) NOT NULL,
	[Address] varchar(60) NOT NULL,
	Email varchar(60) NOT NULL,
	PrimaryPhone varchar(60) NOT NULL,
	CONSTRAINT pk_CustomerId PRIMARY KEY(CustomerId),
	CONSTRAINT fk_ZipCodeCustomer FOREIGN KEY(ZipCode) REFERENCES a_Location(ZipCOde)
);

CREATE TABLE a_Employee(
	EmpId int NOT NULL,
	ZipCode int NOT NULL,
	FirstName varchar(60) NOT NULL,
	LastName varchar(60) NOT NULL,
	[Address] varchar(60) NOT NULL,
	Email varchar(60) NOT NULL,
	PhoneNumber varchar(60) NOT NULL,
	CONSTRAINT pk_EmployeeId PRIMARY KEY(EmpId),
	CONSTRAINT fk_ZipCodeEmp FOREIGN KEY(ZipCode) REFERENCES a_Location(ZipCOde)
);

CREATE TABLE a_Distributor(
	DistributorId int NOT NULL,
	ZipCode int NOT NULL,
	Name varchar(100) NOT NULL,
	[Address] varchar(100) NOT NULL,
	Email varchar(100) NOT NULL,
	PhoneNumber varchar(60) NOT NULL,
	CONSTRAINT pk_DistributorId PRIMARY KEY(DistributorId),
	CONSTRAINT fk_ZipCodeDistributor FOREIGN KEY(ZipCode) REFERENCES a_Location(ZipCOde)
);

CREATE TABLE a_Model(
	ModelNumber int NOT NULL,
	MSRP money Not Null,
	Description varchar (1000),
	CONSTRAINT pk_modelNumber PRIMARY KEY(ModelNumber)
);

CREATE TABLE a_ReasonReturned(
	ReasonReturnID int IDENTITY (1, 1) NOT NULL,
	ReasonDescription varchar(100)
	CONSTRAINT pk_reasonReturnID PRIMARY KEY(ReasonReturnID)
);

CREATE TABLE a_Unit(
	SerialNumber int NOT NULL,
	ModelNumber int NOT NULL,
	DateReturned date,
	CONSTRAINT pk_SerialNumber PRIMARY KEY(SerialNumber),
	CONSTRAINT fk_ModelNumber FOREIGN KEY (ModelNumber) REFERENCES a_Model(ModelNumber),
);

CREATE TABLE a_ReturnReasonSelection(
	ReturnReasonSelectionID int IDENTITY (1, 1) NOT NULL,
	SerialNumber int NOT NULL,
	ReturnReasonID int NOT NULL,
	Comments varchar(1000),
	CONSTRAINT pk_returnReasonSelectionID PRIMARY KEY(ReturnReasonSelectionID),
	CONSTRAINT fk_SerialNumber FOREIGN KEY (SerialNumber) REFERENCES a_Unit(SerialNumber),
	CONSTRAINT fk_ReasonReturnId FOREIGN KEY (ReturnReasonID) REFERENCES a_ReasonReturned(ReasonReturnID)
);
CREATE TABLE a_IssueReportProblem(
	ProblemID int IDENTITY (1, 1) NOT NULL,
	[Description] varchar(100),
	CONSTRAINT pk_ProblemID PRIMARY KEY (ProblemID)
);

CREATE TABLE a_ReportMethod(
    ReportMethodId int IDENTITY (1, 1) NOT NULL,
    Method varchar(60) NOT NULL
    CONSTRAINT pk_reportMethod PRIMARY KEY (ReportMethodId)
);

CREATE TABLE a_IssueReport(
	IssueReportId int IDENTITY (1, 1) NOT NULL,
	ProductSerialNumber int NOT NULL,
	DistributorId int,
	EmployeeId int,
	CustomerId int,
	IssueReportProblemId int NOT NULL,
	ProblemDescription varchar(1000),
	ReportDate date,
	ReportMethodId int NOT NULL,
    InjuryOccurred bit NOT NULL,
    InjuryDescription varchar(1000) NOT NULL,
	CONSTRAINT pk_IssueReportId PRIMARY KEY(IssueReportId),
    CONSTRAINT fk_ReportMethodId FOREIGN KEY(ReportMethodId) REFERENCES a_ReportMethod(ReportMethodId),
	CONSTRAINT fk_ProductSerialNumber FOREIGN KEY(ProductSerialNumber) REFERENCES a_Unit(SerialNumber),
	CONSTRAINT fk_DistributorReportId FOREIGN KEY(DistributorId) REFERENCES a_Distributor(DistributorId),
	CONSTRAINT fk_EmployeeReportId FOREIGN KEY(EmployeeId) REFERENCES a_Employee(EmpId),
	CONSTRAINT fk_CustomerId FOREIGN KEY(CustomerId) REFERENCES a_Customer(CustomerId),
	CONSTRAINT fk_TypeofProblem FOREIGN KEY(IssueReportProblemId) REFERENCES a_IssueReportProblem(ProblemId)
);


CREATE TABLE a_TestForm(
	TestFormId int IDENTITY (1,1) NOT NULL,
	RecommendingTestFormId int,
	IssueReportId int NOT NULL,
	EmpId int NOT NULL,
	TestType varchar(100),
    TestDescription varchar(500) NOT NULL,
	TestResults varchar(1000),
	RecommendedResolution varchar(1000),
	TestCompleted Date,
	CONSTRAINT pk_TestFormId PRIMARY KEY(TestFormId),
	CONSTRAINT fk_RecommendingTestFormId FOREIGN KEY(RecommendingTestFormId) REFERENCES a_TestForm(TestFormId),
	CONSTRAINT fk_IssueReportIdTestForm  FOREIGN KEY(IssueReportId ) REFERENCES a_IssueReport(IssueReportId),
	CONSTRAINT  fk_EmpId FOREIGN KEY(EmpId) REFERENCES a_Employee(EmpId),
);

CREATE TABLE a_UnitLog(
	UnitLogId int IDENTITY(1,1) NOT NULL,
	UserName varchar(30) NOT NULL,
	LogDate date NOT NULL,
	SerialNumber int not NULL,
	CONSTRAINT pk_UnitLogId PRIMARY KEY(UnitLogId),
	CONSTRAINT fk_unitID FOREIGN KEY(SerialNumber) REFERENCES a_unit(serialNumber)
);

CREATE TABLE Employee_a_TestForm_Audit(
	Emp_name varchar(100),
	Audit_Action varchar(100),
	Audit_Timestamp datetime
);
