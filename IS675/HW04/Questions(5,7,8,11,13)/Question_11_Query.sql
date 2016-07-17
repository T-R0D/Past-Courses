-- Summarize the MaterialPurchased in the MaterialPurchased table by the year of the date purchased. 
-- Include the year of the date purchased, the sum of the
-- quantity, a count of the number of POID’s, and the extended purchase cost (quantity * CostPerUOM)


SELECT 
	DISTINCT YEAR(DatePurchased) 'Purchase Year',
	COUNT(*) 'Number of Purchase Orders',
	SUM(Quantity) 'Total Quantity Purchased',
	SUM(Quantity * CostPerUOM) 'Total Amount Purchased'
FROM 
	MaterialPurchased
GROUP BY
	YEAR(DatePurchased);