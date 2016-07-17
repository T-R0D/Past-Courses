	/*Rename simply to be easily recognized*/

rename  areaname County_Area
rename  percentofpersonswithlessthanhigh lessthanHSpercent
rename  percentofpersonswithonlyahighsch onlyHSpercent
rename  percentofpersonscompletingsomeco someCollegepercent
rename  percentofpersonswithacollegedegr collegeDegreepercent
rename  age_19_under under_age19
rename  age_19_64 age19_64
rename  age_65_84 age65_84
rename  age_85_and_over over_age85

	/*Recoding variables to eliminate missing data codes*/

recode obesity (-1111.1=.), gen (Obesity_Rate)
recode no_exercise (-1111.1=.), gen ( no_exerRate)
recode few_fruit_veg (-1111.1=.), gen ( few_fruitvegRate)
recode high_blood_pres (-1111.1=.), gen ( high_BPRate)
recode smoker (-1111.1=.), gen ( smokerRate)
recode diabetes (-1111.1=.), gen ( diabetesRate)
recode elderly_medicare (-2222=.), gen ( elder_medicareRec)
recode disabled_medicare (-2222=.), gen ( disabl_medicareRec)
recode poverty (-2222.2=.), gen ( povertyRec)
recode dentist_rate (-2222.2=.), gen ( dentist_rateRec)
recode pap_smear (-2222.2=.), gen ( Pap_SmearRate)
recode mammogram (-2222.2=.), gen ( MammogrmRate)
recode proctoscopy (-2222.2=.), gen ( ProctoscopRate)
recode unemployed (-9999=.) (-2222=.), gen (unemplRec)
recode sev_work_disabled (-2222=.), gen (sev_work_disablRec)

	/*Recode for usability*/

recode community_health_center_ind (1=0) (2=1), gen (CenterYN)
recode hpsa_ind (1=0) (2=1), gen (HP_ShortageYN)


	/*Generate new rate variables*/

*These variables were counts that needed to be turned into rates*
gen uninsuredRate= (uninsured/ population_size)
gen elder_medcarRate= ( elder_medicareRec/ population_size)
gen disabl_medcarRate= (  disabl_medicareRec/ population_size)
gen unemplRate= (unemplRec/ population_size)
gen maj_depressRate= (major_depression/ population_size)
gen sev_work_disablRate= (sev_work_disablRec/ population_size)

	/*These variables need to be turned into more wieldy rates*/

gen Phys_Rateper100= (prim_care_phys_rate/ 1000)
gen Dent_Rateper100= (dentist_rateRec/ 1000)


