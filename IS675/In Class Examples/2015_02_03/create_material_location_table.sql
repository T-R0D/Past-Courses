DROP TABLE
	xtblMaterialLocation
;

CREATE TABLE
	xtblMaterialLocation(
		  VendorID		char(4)
		, MaterialID	char(3)
		, LocationID	char(3)
		, QuantityOnHand decimal(8,3)
		  CONSTRAINT pkMaterialLocation
			PRIMARY KEY (VendorID, MaterialID, LocationID)
		, CONSTRAINT fkVendorMaterial
			FOREIGN KEY (VendorID, MaterialID)
			REFERENCES xtblVendorMaterial (VendorID, MaterialID)	);INSERT INTO	xtblMaterialLocationVALUES	  ('7189', '255', '12', 700.35)	, ('7189', '255', '15', 600.88)	, ('0062', '271', '81', 5505)	, ('0062', '240', '82', 6);SELECT	*FROM	xtblMaterialLocationALTER TABLE	xtblMaterialLocationDROP	CONSTRAINT		fkVendorMaterial;