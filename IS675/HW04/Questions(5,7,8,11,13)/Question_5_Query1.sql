-- 5. What is the average EstMaterialCostPerSqFt, and EstLaborCostPerSqFt for all tasks completed two years ago? 
-- What was the largest and smallest of those
-- values? Round the result to two digits after the decimal point. Result table: 

SELECT 
	ROUND(AVG(EstMaterialCost/SquareFeet),2) 'Average EstMaterialCostPerSqFt',
	ROUND(MAX(EstMaterialCost/SquareFeet),2) 'Largest EstMaterialCostPerSqFt',
	ROUND(MIN(EstMaterialCost/SquareFeet),2) 'Smallest EstMaterialCostPerSqFt',
	ROUND(AVG(EstLaborCost/SquareFeet),2) 'Average EstLaborCostPerSqFt',
	ROUND(MAX(EstLaborCost/SquareFeet),2) 'Largest EstLaborCostPerSqFt',
	ROUND(MIN(EstLaborCost/SquareFeet),2) 'Smallest EstLaborCostPerSqFt'  
FROM 
	JobTask
WHERE
	DateCompleted is not NULL AND
	DATEDIFF(YEAR, DateCompleted, GETDATE()) = 2;