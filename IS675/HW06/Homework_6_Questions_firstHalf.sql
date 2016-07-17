/*#15 from homework 5*/
create view time_summary as
Select 
	TimeSheet.EmpID,
	Emp.LastName + ','+Emp.FirstName as EmployeeName,
	timeSheet.TaskID,
	TimeSheet.JobID,
	TimeSheet.StartWork,
	TimeSheet.HoursWorked,
	EmpPay.HourlyPayRate as HourlyPay,
	TimeSheet.HoursWorked*EmpPay.HourlyPayRate   as LaborCost
from TimeSheet 
		join Employee as Emp
			on TimeSheet.EmpID = Emp.EmpID
		join EmployeePay as EmpPay
			on TimeSheet.EmpID = EmpPay.EmpID
where (EmpPay.DateStartPay <= TimeSheet.StartWork and TimeSheet.StartWork< isNUll(EmpPay.DateEnd, getdate()))
	  and TimeSheet.JobId is not null; 
select * from time_Summary order by taskID;

--drop view  time_summary

/*1

Summarize the actual hours worked and labor cost (hours worked * hourly pay rate) by JobID and TaskID for all rows in the JobTask table.
Remember that an employee's pay rate changes by date, so it is necessary to locate the correct pay rate by date as you did for question #15 
for HW#5. If you were not able to get question #15 on HW#5 to work, I recommend doing that before starting on this question. 
Compare the actual labor hours to the estimated labor hours, and the actual labor costs to the estimated labor costs. 
The full result table is provided in two pieces below and on the top of the next page.
*/
create view actual_hours as
select
	ts.JobID,
	ts.TaskID,
	sum(ts.HoursWorked) as ActualHoursWorked,
	sum(ts.HoursWorked*ts.HourlyPay) as LaborCost
from time_summary as ts
group by
	ts.JobID,ts.TaskID
;
select* from actual_hours;
--drop view actual_hours;
create view [JobTask_Cost_Est] as
Select
	JobTask.JobID,
	JobTask.TaskID,
	Task.TaskDescription,
	convert(varchar(12),JobTask.DateStarted,101) as DateStarted,
	isNull(convert(varchar(12),JobTask.DateCompleted,101), 'Not Done') as DateCompleted,
	JobTask.EstHours,
	isNull(actual_hours.ActualHoursWorked, 0.00) as ActualHoursWorked,
	(JobTask.EstHours - isNULL(actual_hours.ActualHoursWorked,0.00)) as LaborHoursVariance,
	JobTask.EstLaborCost,
	isNULL(actual_hours.LaborCost,0.00) as LaborCost,
	(JobTask.EstLaborCost - isNULL(actual_hours.LaborCost,0.00)) as LaborCostVariance 
from JobTask
	join Task on JobTask.TaskID = Task.TaskID
	left join actual_hours
			on JobTask.JobID = actual_hours.JobID
			and JobTask.TaskID = actual_hours.TaskID;		
select * 
	from [JobTask_Cost_Est] as JTCE
	order by JTCE.JobID, JTCE.TaskID;
drop view [JobTask_Cost_Est]
/*2
Summarize the actual material costs by jobID and task ID and compare them to the 
estimated material cost for each row in the JobTask table.  This is very similar to what you did for question #11 in HW#5, 
so this should be fairly easy if you got question #11 to work.  The result table is divided and shown below and on the next page:    
*/
create view JobTask_Task_summary as
select 
	JobTask.JobID,
	JobTask.TaskID,
	Task.TaskDescription,
	convert(varchar(12),JobTask.DateStarted,101) as DateStarted,
	isNULL(convert(varchar(12),JobTask.DateCompleted,101),'Not Done') as DateCompleted,
	JobTask.EstMaterialCost
from JobTask
	left join Task
	on JobTask.TaskID = Task.TaskID;

--select *from JobTask_Task_summary as JTT order by JTT.JobID, JTT.TaskID; 
--drop view JobTask_Task_summary;

create view Material_Ass_Pur as
Select 
	MA.JobID,
	MA.TaskID,
	sum(MA.Quantity*MP.CostPerUOM) as ActualMaterialCost
from MaterialAssigned as MA
	left join MaterialPurchased as MP
	on MA.POID = MP.POID
group by MA.JobID, MA.TaskID;

--drop view Material_Ass_Pur;
create view [merge_JTT_MAP] 
as
select 
	JTT.JobID,
	JTT.TaskID,
	JTT.TaskDescription,
	JTT.DateStarted,
	JTT.DateCompleted,
	JTT.EstMaterialCost,
	Cast(isNULL(MAP.ActualMaterialCost,0.00) as money) as ActualMaterialCost,
	Cast (isNULL((JTT.EstMaterialCost - MAP.ActualMaterialCost),0.00) as Money) 
	as MaterialCostVariance
from JobTask_Task_summary as JTT
	left join Material_Ass_Pur as MAP
	on JTT.JobID=MAP.JobID and JTT.taskID = MAP.TaskID;

select * from [merge_JTT_MAP] 
order by [merge_JTT_MAP] .JobID, [merge_JTT_MAP] .TaskID;

--drop view [merge_JTT_MAP];
/*3 
3.  Now it’s time to put them together.  
Compare actual to estimated costs for each row in the JobTask table.  
The PercentVariance is the percentage variance between the TotalEstCost and the TotalActualCost.  
The general calculation is:  ((TotalEstCost – TotalActualCost)/TotalEstCost) * 100.  
Virtually unreadable result table provided below.

*/
create view conundrum as 
select 
	JTCE.JobID,
	JTCE.TaskID,
	JTCE.TaskDescription,
	JTCE.DateStarted,
	isNULL(JTCE.DateCompleted,'Not Done') as DateCompleted,
	JTCE.EstHours,
	JTCE.ActualHoursWorked,
	JTCE.LaborHoursVariance,
	JTCE.EstLaborCost,
	JTCE.LaborCost as ActualLaborCost,
	JTCE.LaborCostVariance,
	JTTMAP.EstMaterialCost,
	JTTMAP.ActualMaterialCost,
	JTTMAP.MaterialCostVariance,
	(JTTMAP.EstMaterialCost+JTCE.EstLaborCost) as TotalEstCost,
	(JTTMAP.ActualMaterialCost+JTCE.LaborCost) as TotalActualCost,
	((JTTMAP.EstMaterialCost+JTCE.EstLaborCost) 
	  -
	(JTTMAP.ActualMaterialCost+JTCE.LaborCost))  as TotalCostVariance,
	((((JTTMAP.EstMaterialCost+JTCE.EstLaborCost) 
	  -
	(JTTMAP.ActualMaterialCost+JTCE.LaborCost))
	/
	(JTTMAP.EstMaterialCost+JTCE.EstLaborCost))
	* 100)
	as PercentVariance
from [JobTask_Cost_Est] as JTCE
	join [merge_JTT_MAP] as JTTMAP
		on JTCE.jobID = JTTMAP.jobID
			and JTCE.TaskID = JTTMAP.TaskID;

select * from conundrum as cd
order by cd.JobID, cd.TaskID;
	
--drop view conundrum;

/*4
Summarize the information created in question #3 by job. The result table should have one row per job in the Job table.  
Add additional data from the Job and Client tables to provide more information about each job in the result table.  
Hint:  Calculate the PercentVariance – you cannot sum that field.   
*/
create view [job_client] as
select 
	job.JobID,
	Job.JobName,
	client.ClientName,
	(case 
		when job.JobCompleted = 0
		then 'Not Finshed'
		else 'Finished'
	end) as JobStatus
from job
	join Client 
	on job.ClientID = client.ClientID;

select * from [job_client] as jc
order by jc.JobID;

--drop view [job_client];
create view [sumarized_info_by_job] as
select
	jc.JobID,
	jc.JobName,
	jc.jobStatus,
	sum(cd.EstHours) as EstHours,
	sum(cd.ActualHoursWorked) as ActualHoursWorked,
	sum(cd.LaborHoursVariance) as LaborHoursVariance,
	sum(cd.EstLaborCost) as EstLaborCost,
	sum(cd.ActualLaborCost) as ActualLaborCost,
	sum(cd.LaborCostVariance) as LaborCostVariance,
	sum(cd.EstMaterialCost) as EstMaterialCost,
	sum(cd.ActualMaterialCost) as ActualMaterialCost,
	sum(cd.MaterialCostVariance) as MaterialCostVariance,
	sum(cd.TotalEstCost) as TotalEstCost,
	sum(cd.TotalActualCost) as TotalActualCost,
	sum(cd.TotalCostVariance) as TotalCostVariance,
	((sum(cd.TotalCostVariance)/sum(cd.TotalEstCost))*100) as PercentVariance
from [job_client] as jc
left join conundrum as cd
on jc.JobID = cd.JobID
group by jc.JobID,jc.JobName,
	jc.jobStatus
;
select* from  [sumarized_info_by_job];

--drop view  [sumarized_info_by_job];
/* 5
Which job that is finished had actual total costs that were closest to the estimated total costs? 
(PercentVariance closest to zero)  Make sure that the query could select the correct job from any data set 
– the query should not just work with our test data set.  Result table: 
*/
select *
from [sumarized_info_by_job] as sum_job
where sum_job.PercentVariance = (select min (abs(sj2.PercentVariance)) 
						from [sumarized_info_by_job] as sj2
						 );


/*6
Which job that is finished had the largest percentage positive labor hours variance? In other words, which finished 
job was able to be completed with the least number of labor hours, when compared to the estimated labor hours?  
The percentage labor hours variance is calculated as the LaborHoursVariance/EstHours * 100. 
 Add in the name of the employee who served as the manager for the job.  Result table: 
*/
create view [largest_percent_labor_hours_variance] as
select
	jc.JobID,
	jc.JobName,
	jc.ClientName,
	sj.EstHours,
	sj.ActualHoursWorked,
	sj.LaborHoursVariance,
	((sj.LaborHoursVariance/sj.EstHours)*100) as PercentHoursVariance
from job_client as jc
	 left join sumarized_info_by_job as sj
	 on jc.JobID = sj.JobID
where jc.JobStatus = 'Finished'
;

select* from [largest_percent_labor_hours_variance] as maxLaborVar
	where maxLaborVar.PercentHoursVariance = (select max(maxLaborVar2.PercentHoursVariance) 
	 from [largest_percent_labor_hours_variance]  as maxLaborVar2)
;
--drop view [largest_percent_labor_hours_variance];

/*7
What is the average amount of time (labor hours) spent on a completed job task per square foot, 
as compared to the estimated amount of time that should be spent on a task per square foot

Use the data in the JobTask table to calculate the average amount of EstHours/Squarefeet, but use the data 
in the TimeSheet table to calculate the average amount of time that was actually worked on a completed task. 
I recommend creating separate views for the estimated hours per square feet and the actual hours per square feet.  
The estimate view is a little easier to create because it doesn’t require a join.  Include all rows in the JobTask
table to get the average EstHours/Squarefeet for a task.  To get the average actual hours per square feet requires
that you join the TimeSheet table and the JobTask table to be able to use the square feet in the JobTask table.  
Do not include data for incompleted tasks when calculating the ActualHours/SquareFeet.  Remember that you have to 
SUM the HoursWorked in the TimeSheet table by JobID and TaskID to get the Actual HoursWorked from the TimeSheet table.   
I rounded the final results to 6 digits after the decimal point.  The result table is at the top of the next page.  
There is one row in the result table for each row in the Task table.  Sort the result table by TaskID. 
*/


/*8
Use the result table generated for question #7 to help you answer this question.  
The goal of this query is to identify which task has the largest negative difference between the 
EstimatedHoursPerSqFt and ActualHoursPerSqFt (which estimate is the worst because the actual is larger).  
*/

/*9
The objective of this query is similar to that for question #7, except this time we are going to look at labor costs rather than labor hours.  
What is the average estimated labor cost per square foot as compared to the actual labor cost per square foot for each task?  
I recommend looking back at question #1, where you probably created a view to help you calculate actual labor costs for a task on a job.  
That view will help you with this question.  Do not include data for incompleted tasks when calculating the actual labor cost/SquareFeet;
do include data for incompleted tasks when calculating the estimated labor cost/squarefeet.
*/


/*10
Which clients did not have any jobs with a DateAccepted last year?  
Which materials were not assigned (DateAssigned) to any job tasks last year?  
Combine the clients and materials into a single result table (hint:  Use the UNION statement). 
Make sure that you use the GETDATE() function to determine the correct year. 
*/

--drop view Material_Ass_Pur; --view that merges data from material purchased and material assigned
--drop view  time_summary;   --view that sumarizes all timesheet data from problem 15 homework 5 (employee hrs worked, payrate and labor cost based on job and task)   
--drop view actual_hours;   -- view that sumarizes data from time_summary view and computes the acutal hours and laborCost
--drop view JobTask_Task_summary; --view that computes the est material costs per job and task and includes the task description as well
--drop view [merge_JTT_MAP];   --view that takes info from [JobTask_Task_summary]  and Material_Ass_Pur (sumarizes est material cost, actual material cost and material cost var)
--drop view [JobTask_Cost_Est] -- view that sumarizes data for each job (job info and est labor and hour cost and variances)
--drop view conundrum - view that merges info from jobTask,Material,MaterialPurchased and Assigned, and comptues TotalEstCost, Total ActualCost, Total Cost Variance and Percent Variance
--drop view [job_client]; -- view that merges information from job and Client to get JobID, Name, ClientName and JobStatus
--drop view  [sumarized_info_by_job]; --view that merges info from conundrum(all info for jobs and tasks) and job_client to summarize all info based on just the JobID (got rid of TaskID)
--drop view [largest_percent_labor_hours_variance] --view that merges job_client and conundrum and summarizes (EstHours, Actual Work/Var/%Var for each job(gets rid of task)


select* from  [largest_percent_labor_hours_variance];