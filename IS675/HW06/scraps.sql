--3
SELECT
    [JobTask].[JobID] AS 'JobID',
    [JobTask].[TaskID] AS 'TaskID',
    [Task].[TaskDescription] AS 'Task Description',
    CONVERT(varchar, [JobTask].[DateStarted], 101) AS 'DateStarted',
    ISNULL(
        CONVERT(varchar, [JobTask].[DateCompleted], 101), 'Not Done'
    ) AS 'DateCompleted',
    [v_LaborCostComparisons].[EstHours] AS 'EstHours',
    [v_LaborCostComparisons].[ActualHoursWorked] AS 'ActualHoursWorked',
    (
        [v_LaborCostComparisons].[ActualHoursWorked] -
        [v_LaborCostComparisons].[EstHours]
    ) AS 'LaborHoursVariance',
    [JobTask].[EstLaborCost] AS 'EstLaborCost',
    ISNULL([v_ActualLaborByTask].[LaborCost], 0.00) AS 'ActualLaborCost',
    (
        [JobTask].[EstLaborCost] -
        ISNULL([v_ActualLaborByTask].[LaborCost], 0.00)
    ) AS 'LaborCostVariance',
    [JobTask].[EstMaterialCost] AS 'EstMaterialCost',
    ISNULL(
        [v_ActualMaterialsByTask].[MaterialCost], 0.00
    ) AS 'ActualMaterialCost',
    (
        [JobTask].[EstMaterialCost] -
        ISNULL([v_ActualMaterialsByTask].[MaterialCost], 0.00)
    ) AS 'MaterialCostVariance',
    [JobTask].[EstLaborCost] + [JobTask].[EstMaterialCost] AS 'TotalEstCost',
    (
        ISNULL([v_ActualLaborByTask].[LaborCost], 0.00) +
        ISNULL([v_ActualMaterialsByTask].[MaterialCost], 0.00)
    ) AS 'TotalActualCost',
    (
        ([JobTask].[EstLaborCost] + [JobTask].[EstMaterialCost]) -
        ISNULL(
            [v_ActualLaborByTask].[LaborCost] + [v_ActualMaterialsByTask].[MaterialCost],
            0.00
        )
    ) AS 'TotalCostVariance',
    (
        ([JobTask].[EstLaborCost] + [JobTask].[EstMaterialCost]) -
        (
            ISNULL([v_ActualLaborByTask].[LaborCost], 0.00) +
            ISNULL([v_ActualMaterialsByTask].[MaterialCost], 0.00)
        )
    ) * 100.00 / ([JobTask].[EstLaborCost] + [JobTask].[EstMaterialCost])
        AS 'PercentVariance'
FROM
    [JobTask]
        INNER JOIN [Task] ON
            [JobTask].[TaskID] = [Task].[TaskID]
        INNER JOIN [v_LaborCostComparisons] ON
            [JobTask].[JobID] = [v_LaborCostComparisons].[JobID] AND
            [JobTask].[TaskID] = [v_LaborCostComparisons].[TaskID]
        INNER JOIN [v_MaterialCostComparisons] ON
            [JobTask].[JobID] = [v_MaterialCostComparisons].[JobID] AND
            [JobTask].[TaskID] = [v_MaterialCostComparisons].[TaskID]
ORDER BY
    [JobTask].[JobID],
    [JobTask].[TaskID]
;