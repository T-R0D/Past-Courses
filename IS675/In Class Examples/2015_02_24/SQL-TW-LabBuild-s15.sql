
DROP TABLE ztbltimeworked;
DROP TABLE ztblemp;
DROP TABLE ztblcontract;
DROP TABLE ztblclient;
DROP TABLE ztbljobtitle;

CREATE TABLE ztblJobTitle
(jobtitleid   char(2),
 title        varchar(50));


CREATE TABLE ztblEmp
(empid        	char(5) primary key,
 lastname          	varchar(20)	NOT NULL,
 firstname		varchar(20),
 address		varchar(30),
 city	     		varchar(20),
state			char(2),
zip			char(9),
hiredate		datetime				NOT NULL,
officephone	char(10)				NOT NULL,
billingrate	money				NOT NULL,
managerID		char(5),
jobtitleID	char(2));


CREATE TABLE	ztblClient
(clientid		char(4) primary key,
name			varchar(40),
address		varchar(30),
city			varchar(15),
state			char(2),
zip			char(9),
phone			char(10),
faztbl			char(10),
contactname		varchar(20));


CREATE TABLE	ztblContract
(contractid  	char(4) primary key,
 Clientid    	char(4),
 datesigned		datetime,
 datedue		datetime

);



 
CREATE TABLE  ztblTimeWorked
(
TimeWorkedID int identity primary key, 
EmpID        char(5),
 startwork    datetime,
 ContractID   char(4),
 Worktypeid   char(3),
 minutes       int,
 
);


INSERT INTO ztblJOBTITLE VALUES
(20, 'Business Analyst');
INSERT INTO ztbljobtitle VALUES
(10, 'Manager');
INSERT INTO ztbljobtitle VALUES
(40, 'Database Designer');
INSERT INTO ztbljobtitle VALUES
(45, 'Web Programmer');
INSERT INTO ztbljobtitle VALUES
(50, 'Interface Programmer');
INSERT INTO ztbljobtitle VALUES
(55, 'Graphics Designer');
INSERT INTO ztbljobtitle VALUES
(57, 'SAP Analyst');




INSERT INTO ztblEmp VALUES 
(7819,	'Martinson', 'Cassandra', '123 Main St.',	'Reno',	'nv',	'89557',	'06-oct-2012',	7757812334,	125, null,10);

insert into ztblEmp values
('4522', 'Jenkins', 'Martin', '23 scatterwood', 'Reno', 'nv', '89508', '14-jan-2012', '7754639001', 90, '3424',40);

insert into ztblEmp values
('5633', 'Flanders', 'Janice', 'P.O. Boztbl 445', 'Sparks', 'Nv', '89431', '12-feb-2007', '7756351441', 65, '3424',45) ;

insert into ztblEmp values
('7369', 'Chu', 'Joyce',  '673 Brinkley Lane', 'Las Vegas', 'NV', '89111', '23-aug-2013', '7023456771', 95, '7819',45);

insert into ztblEmp values
('3411', 'Tristan', 'Elliott',  '344 Crestview Ct.', 'RENO', 'NV', '89502', '25-may-2010', '7757843221', 225.00, '7819',55);

insert into ztblEmp values
('3424', 'Polanski', 'Charles', '5662 Caminito Corriente', 'San Diego', 'ca', '92128', '12-jan-2014', '8582339001', 95, '7819', 10);

insert into ztblEmp values
('2412', 'Perez', 'Martina',  '8372 Via Coronado', 'San diego', 'Ca', '92126', '20-dec-2012', '8582123848', 89, '3424', 22);

insert into ztblEmp values
('7715', 'Kendall', 'Brent',  '662 westwood', 'san diego', 'CA', '92128', '12-jan-2014', '5802389912', 75, '3424',50);


insert into ztblClient values
('1200', 'Cancer Research Society', '76 Bell Parkway', 'las vegas', 'nv', '89661', '7023145617', '7027726003', 'mark jones');

insert into ztblClient values
('5600', 'Sobret Manufacturing', '661 Terminal Way', 'reno', 'NV', '89523', '7752215661', '7757845661', 'candice guest');

insert into ztblClient values
('4900', 'Best Industries', '214 Arrow Dr.', 'reno', 'nv', '89510', '7754456771', '7758339001', 'Nancy Ng');

insert into ztblClient values
('5700', 'Springwater Hotel/Casino', '778 Springwater Drive', 'san diego', 'ca', '92124', '8584556991', '8586507985', 'Lori Mosqueda');

insert into ztblClient values
('8900', 'Coldcreek Cavern Club', '781 Coldwater St.', 'san diego', 'ca', '92128', '8586433554', '8586738992', 'Bill Worth');

insert into ztblClient values
('9700', 'Crestwave Supply Co.', '562 S. Park Place.', 'san diego', 'ca', '92126', '8583667889', '8587668993', 'Fred Lane');

insert into ztblContract values
('777', '9700', '08-dec-2014','15-mar-2015');

insert into ztblContract values
('333', '9700', '10-feb-2015','18-may-2015');

insert into ztblContract values
('444', '1200', '12-feb-2015','01-jun-2015');

insert into ztblContract values
('666', '4900', '02-jan-2015','22-feb-2015');

insert into ztblContract values
('662', '4900', '13-feb-2015','01-may-2015');

insert into ztblContract values
('555', '5600', '10-feb-2015', '15-jun-2015');

insert into ztblContract values
('781', '5700', '12-dec-2014', '02-mar-2015');




INSERT INTO ztblTimeWorked VALUES
(7819, '06-feb-2015 4PM', 777,	455, 300);

INSERT INTO ztblTimeWorked VALUES
(7819,	'06-feb-2015 8AM', 777,	255, 100);


insert into ztblTimeWorked values
('7819', '15-jan-2015 9AM', 777, '455', 300);

insert into ztblTimeWorked values
('7819', '16-jan-2015 9AM', 777, '455', 100);

insert into ztblTimeWorked values
('7819', '28-jan-2015 8AM', 666, '255', 120);

insert into ztblTimeWorked values
('5633', '02-jan-2015 8AM', '777', '455', 480);

insert into ztblTimeWorked values
('5633', '03-jan-2015 8AM', '666', '455', 600);

insert into ztblTimeWorked values
('5633', '04-jan-2015 8AM', '666', '255', 600);

insert into ztblTimeWorked values
('5633', '05-jan-2015 8AM', '666', '255', 600);

insert into ztblTimeWorked values
('5633', '19-feb-2015 8AM', '666', '003', 450);

insert into ztblTimeWorked values
('4522', '09-feb-2015 8AM', '444', '255', 720);

insert into ztblTimeWorked values
('3411', '12-feb-2015 8AM', '444', '003', 700);

insert into ztblTimeWorked values
('3411', '08-feb-2015 8AM', '444', '451', 480);

insert into ztblTimeWorked values
('3411', '08-feb-2015 6PM', '444', '003', 120);

insert into ztblTimeWorked values
('3424', '08-feb-2015 8AM', '444', '451', 480);

insert into ztblTimeWorked values
('3424', '09-feb-2015 2PM', '444', '003', 180);

insert into ztblTimeWorked values
('4522', '02-feb-2015 8AM', 777, '451', 120);

insert into ztblTimeWorked values
('3424', '03-feb-2015 8AM', 777, '003', 480);

insert into ztblTimeWorked values
('7369','10-feb-2015 8AM', 666, '451', 480);

insert into ztblTimeWorked values
('7369', '06-feb-2015 8AM', 777, 451, 120);

insert into ztblTimeWorked values
('7369', '10-feb-2015 3PM', 666, 451, 120);

insert into ztblTimeWorked values
('4522', '27-feb-2015 8AM', 666, '003', 180);

insert into ztblTimeWorked values
('4522', '27-feb-2015 11AM', 777, '003', 220);

insert into ztblTimeWorked values
('4522', '27-feb-2015 3PM', 666, 451, 120);

insert into ztblTimeWorked values
('4522', '28-feb-2015 8AM', 777, 255, 480);

insert into ztblTimeWorked values
('4522', '10-jan-2015 8AM', 555, 480, 540);

insert into ztblTimeWorked values
('4522', '11-jan-2015 8AM', 555, 480, 600);

insert into ztblTimeWorked values
('4522', '12-jan-2015 8AM', 555, 480, 600);

insert into ztblTimeWorked values
('4522', '13-jan-2015 8AM', 662, 480, 600);

insert into ztblTimeWorked values
('4522', '14-jan-2015 8AM', 662, 480, 600);

insert into ztblTimeWorked values
('7369', '10-jan-2015 8AM', 555, 461, 540);

insert into ztblTimeWorked values
('7369', '11-jan-2015 8AM', 555, 461, 600);

insert into ztblTimeWorked values
('7369', '12-jan-2015 8AM', 555, 461, 600);

insert into ztblTimeWorked values
('7369', '13-jan-2015 8AM', 662, 461, 600);

insert into ztblTimeWorked values
('7369', '14-jan-2015 8AM', 662, 461, 600);

insert into ztblTimeWorked values
('3424', '10-jan-2015 9AM', 662, 461, 120);


--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SELECT
	tw.EmpID,
	tw.ContractID,
	sum(minutes/60) TotalHoursWorked
FROM
	ztblTimeWorked AS tw
GROUP BY
	tw.EmpID,
	tw.ContractID
ORDER BY
	tw.EmpID
;


SELECT
	tw.EmpID,
	lastName + ', ' + firstname AS EmployeeName,
	tw.ContractID,
	sum(minutes/60) TotalHoursWorked
FROM
	ztblTimeWorked AS tw
		INNER JOIN ztblEmp AS emp ON
			tw.EmpID = emp.EmpID
GROUP BY
	tw.EmpID,
	tw.ContractID,
	lastname + ', ' + firstname
ORDER BY
	tw.EmpID
;

SELECT
	tw.EmpID,
	lastName + ', ' + firstname AS EmployeeName,
	tw.ContractID,
	sum(minutes/60) TotalHoursWorked
FROM
	ztblTimeWorked AS tw
		INNER JOIN ztblEmp AS emp ON
			tw.EmpID = emp.EmpID
GROUP BY
	tw.EmpID,
	tw.ContractID,
	lastname + ', ' + firstname
ORDER BY
	tw.EmpID
;


SELECT
	tw.EmpID,
	lastName + ', ' + firstname AS EmployeeName,
	jt.Title JobTitle,
	tw.ContractID,
	sum(minutes/60) TotalHoursWorked
FROM
	ztblTimeWorked AS tw
		INNER JOIN ztblEmp AS emp ON
			tw.EmpID = emp.EmpID
		LEFT OUTER JOIN ztblJobTitle AS jt ON
			jt.jobtitleID = emp.jobtitleID
GROUP BY
	tw.EmpID,
	tw.ContractID,
	lastname + ', ' + firstname,
	jt.Title
ORDER BY
	tw.EmpID
;


SELECT
	tw.EmpID,
	lastName + ', ' + firstname AS EmployeeName,
	jt.Title JobTitle,
	tw.ContractID,
	client.name ClientName,
	sum(minutes/60) TotalHoursWorked
FROM
	ztblTimeWorked AS tw
		INNER JOIN ztblEmp AS emp ON
			tw.EmpID = emp.EmpID
		LEFT OUTER JOIN ztblJobTitle AS jt ON
			jt.jobtitleID = emp.jobtitleID
		INNER JOIN ztblContract AS [contract] ON
			[contract].contractid = tw.ContractID
		INNER JOIN ztblClient AS client ON
			client.clientid = [contract].Clientid
GROUP BY
	tw.EmpID,
	tw.ContractID,
	lastname + ', ' + firstname,
	jt.Title,
	client.name
ORDER BY
	tw.EmpID
;


SELECT
	jobtitleid,
	AVG(billingrate)
FROM
	ztblemp
GROUP BY
	jobtitleID
;


CREATE VIEW vAvgRateByTitle AS
SELECT
	jobTitleID,
	AVG(billingrate) AvgBillingRate
FROM
	ztblEmp
GROUP BY
	jobtitleID
;

SELECT
	*
FROM
	vAvgRateByTitle
;