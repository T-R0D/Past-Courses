DROP TABLE
	xtblVendorMaterial
;

CREATE TABLE
	xtblVendorMaterial (
		  VendorID		char(4)
		, MaterialID	char(3)
		, CurrentPrice	money		NOT NULL
		  CONSTRAINT pkVendorMaterial
			PRIMARY KEY (VendorID, MaterialID)
		, CONSTRAINT fkVendor
			FOREIGN KEY (VendorID) REFERENCES xtblVendor (VendorID)
		, CONSTRAINT fkRM
			FOREIGN KEY (MaterialID) REFERENCES xtblRawMaterial (MaterialID)
	)
;

INSERT INTO
	xtblVendorMaterial
VALUES
	  ('7189', '255', 14.25)
	, (  '62', '255', 13.95)
	, ('0062', '271', 46.70)
	, ('7189', '240',  0.26)
;

SELECT
	*
FROM
	xtblVendor
;