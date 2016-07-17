/* ||||||||||||||||||||||||||||||||||||||||||||| */
/*            Copyright (c) Jonathan Kelley 2004-2012 */
/*Program eval4    
   USES: hat_all, fix_dummy, fix_vars
   The key change from Eval3 is adding subroutine fix_vars
*/

capture program drop eval4   /*Drop previous version, so can replace */
    *eval4  linear   faoccs4q   "0 33 67 100"  pntEd   "9 12 15"
    *       1=type   2=newvar   3=values       4=var2  5=values2
program define eval4 
     args type newvar  values var2 values2 
     *these are local macros; you need the wierd single quotes to use them

version 9      /* Just for drill*/
         *display "EVAL echos:"
         *display " `1' " 
         *display " `2' " 
         *display " `3' " 
         *display " `4' " 
         *display " `5' " 
         *display "Entry 1=   `type' "
         *summarize `newvar'
         *display "Entry 3=   `values' "
         *summarize `var2'
         *display "Entry 5=   `values2' "


*----------Initialize
*--- Just in case there are temporary variables
   capture drop t_*                 /*get rid of any temporary vars */


*--- Fancy footwork for when there is no 2nd loop - we need to get a fake
*    entry in the 2nd loop so it runs once rather than zero times.
if "`values2'" == "" {
    *display "Initially we have no 2nd var"
    *Make a dummy entry in values2 so we can get one run of the outter loop
    local values2 = -999999
    *display "2nd set of values are: "
    *display "`values2'" 
    }
else {
    *display "2nd variable is " 
    *summarize `var2'
    *display "2nd set of values are: "
    *display "`values2'" 
    }


*---Initialize sequence in the program (1st loop, 2nd loop, 3d loop...)
*   Will re-initialize it later, in the inner loop. But fix_dummys needs
*   it when 1st called, which is before the inner loop starts.
global eval_count = 0
*There is a check that dummy variables are OK; that is done later, in fix_vars.

*Type needs to be global, so subroutines can see it
global eval_type=  "`type'" 
      *display "Global storage of type is:  $eval_type"

preserve       /*save unchanged copy of data */      

*#0 Echo inputs & error check.

display " "
display as text "*---------------------------------------------------------------------------------*" 

*---Echo.
display "WHOLE POPULATION STANDARDIZATION. Version Jan 2009. For help: eval4 help"



if  "`type'" == "help"    {
**   display "Help for eval4 "
display "   Run a probit, logit, or any linear model (OLS, xtreg, or anything"
display "   for which Stata's predict function works properly with option xb)." 
display "   Choose: linear, logit, probit. Loop over 1 variable, or optionally 2."
display "   Assumes the whole population should be analyzed (otherwise "
display "   analyze a work file restricted to the relevant cases)."
display " "
display `"SIMPLE EXAMPLES "'
display `"      eval4 linear age "18 50 64" "'
display `"      eval4 probit churchGo "ln(0.5) ln(52)" "' 
display `"      eval4 logit age "18 64" status "0 30 35 74 100" "'  
display ""
display "SPECIAL CASES: Interactions, quadratics, sets of dummies, recodes: "
display "   that is whenever one variable depends by definition on others. "
display "   If you want to run Eval4 on them, you need to modify "
display "   subroutine fix_vars (ignore it otherwise)."
display "   fix_vars tells how to compute one variable from the others."
display "   --Interactions, quadratics, and the like must be specified."
display "   --Single dummies (e.g. male, Protestant) are OK as is; do nothing."
display "   --SETS of 2+ dummy variables (e.g. single, married, divorced, other)" 
display "   must be exhaustive, scored 0 or 1, and all listed in fix_vars. "
display "   (but use only those you want in the analysis). "
display "   ---Variables not in the current analysis can still be in fix_vars."
display ""
display  "   EXAMPLE using fix_vars:" 
display `"   capture program drop fix_vars "' 
display `"   program define fix_vars "'
display `"      fix_dummy "single married divorced other"  "'
display `"      fix_dummy "Catholic Protestant Jewish OtherRelig None" "'
display `"      fix_dummy "nsw vic qld sa wa tas regionNEC"  "'
display `"      quietly replace cathXsingle = Catholic * single  "'
display `"      quietly replace status2 = status * status "'
display `"      quietly replace WWII = yearBorn
display `"      quietly recode WWII (1939 / 1945 =1)(* =0)
display `"   end "'
display " "
display  "   probit abort Catholic Protestant Jewish single divorced cathXsingle"
display  "      status status2 yearBorn WWII"
display `"   eval4 probit single "0 1" Catholic "0 1" "'
display `"      [that shows the interaction between Catholic and single] "'
display `"   eval4 probit status "0 10 20 30 40 50 60 70 80 90 100"   "'
display `"      [that shows the curvature in status] "'
display `"   eval4 probit yearBorn "1925 1930 1935 1940 1945 1950 1955 1960" "'
display `"      [that shows the WWII effect] "'
display ""
display  "OUTPUT" 
display "      Now-prev = Difference, predicted value now minus value in previous step"
display "      Now-begin = Difference, predicted value now minus value in very first step"
display "      Now/begin = Ratio, predicted value now to value in very first step"
display ""
display "TYPOLOGIES with more than 2 variables "
display "   Trick: estimate; preserve data; set typology; eval4; restore data:"
display "   Example: "
display "      probit lifesat2 cath lnattend incscaled  gdp_pp3r "
display "      preserve "
display "      replace gdp_pp3r = 1 "
display "      replace lnattend = 3.95 "
display `"      eval4 probit incscaled ".1 .5 1" cath "0 1 "  "'
display "      restore "


display "*---------------------------------------------------------------------------------*" 
exit
} 


*---Error check: probit, logit, or linear .
if  "`type'" == "probit"    {
       display "      This is a probit model."
   }
   else if "`type'" == "logit" {
       display "      This is a logit model."
   }
   else if "`type'" == "linear" {
       display "      This is a linear model."
   }
   else {
       display "Oops! Is this a linear, logit, or probit model?"
       display "The last argument should say which. "
       display "You said ||`type'|| "
       display "For examples: eval4 help"
       exit 
   }


*---- Check for errors in fix_vars 
*---- Also checks that the dummy variables (if any) are OK 
display "      I'll call fix_vars, just to be sure it runs OK (it checks dummies too)"
fix_vars    /* Fix interactions, quadratics, dummies, etc */
display "      Fix_vars ran OK"
display ""


*Main loop

foreach y in `values2' {     /* outter loop (may not be real)*/
    *display "this is the loop we are on now, y=: `y'  and values2 is: " values2
    *-999999 means there is only real one variable, not two

    if `y' != -999999  {
       quietly replace `var2' = `y' 
       display ""
       display as text "*---------- `var2' = `y' " 
    }


*---re-initialize sequence in the program (1st loop, 2nd loop, 3d loop...)
global eval_count = 0

*---initialize predicted value for 1st step of loop (for hat_lin & hat_pbt)
global eval_pred1 = 0 

   foreach x in `values' {
       global eval_count = $eval_count + 1
       *display "Loop count: " $eval_count


    if $eval_count == 1  {      /*header row for the output */
       display as text %10s "`newvar'" %10s "Predicted"  %10s "s.e."  %10s "ci Low" %10s "ci High" %10s "Now-prev" %10s "Now-begin" %10s "Now/begin" 
    }
       
       quietly replace `newvar'=`x'
       fix_vars    /* Fix interactions and quadratics */

       hat_all     /*one subroutine does liner, probit, & logit */
       *---Print main results
             *fixed decimal alternative:  global eval_fmt = "%10.4f"
             *The "g" format takes care of very big (or small) numbers.
       global eval_fmt = "%10.4g"
       display as text %10s $eval_fmt `x' as result $eval_fmt $eval_pred $eval_fmt $eval_se $eval_fmt $eval_ciLow $eval_fmt $eval_ciHi $eval_fmt $eval_change1 $eval_fmt $eval_change  $eval_fmt  $eval_ratio 
      }

  }

*Clean up & exit

restore             /* restore original data */
macro drop eval_*   /* drop globals used in the program */
display as text "*---------------------------------------------------------------------------------*" 

end
/* ||||||||||||||||||||||||||||||||||||||||||||| */






*Subroutine hat_all 

/* ||||||||||||||||||||||||||||||||||||||||||||| 
   Program hat_all : predicted values  (whole pop standardization)  
   USES: Nothing
   CALLED BY: Eval
   Creates vars t_p t_pstd t_plow t_phi  (which need to be killed later) 
*/
   capture program drop hat_all        /*Drop previous version, so can replace */
program define hat_all 
   *display "Subroutine hat_all called"
   version 9      /* norm() is normal() in Stata 10 & old version does not work*/

   capture drop t_*                 /*get rid of any temporary vars */


*-------- get linear "xb" predictions from the index function
quietly predict t_xb, xb           /* linear prediction  */
quietly predict t_xbstd, stdp      /* std error of predicted mean, linear */
quietly gen t_xblow = t_xb - 1.96*t_xbstd    /*lower bound of CI, linear model */
quietly gen t_xbhi  = t_xb + 1.96*t_xbstd    /*upper bound of CI, linear model */
        /*careful: the 1.96 is used below for the se. */


*-------- Link function for probit & logit (do nothing for linear)

if  "$eval_type" == "probit"    {
       *display " hat_all called for a probit model."
    quietly gen t_p = norm(t_xb)       /* prob, from the index function */
    quietly gen t_plow = norm(t_xblow) /* lower CI, probability model  */
    quietly gen t_phi = norm(t_xbhi)   /* higher CI, probability model  */
   }
   else if "$eval_type"  == "logit" {
       *display "  hat_all called for a logit model."
    quietly gen t_p =   1/(1 + exp(-1* t_xb))   /* prob, from the index function */
    quietly gen t_plow = 1/(1 + exp(-1* t_xblow))  /* lower CI, prob model  */
    quietly gen t_phi =  1/(1 + exp(-1* t_xbhi))   /* higher CI, proba model  */
   }
   else if "$eval_type"  == "linear" {
       *display "  hat_all called for a linear model."
    quietly gen t_p = t_xb       /* just renaming, for the output */
    quietly gen t_plow = t_xblow /* just renaming, for the output  */
    quietly gen t_phi = t_xbhi   /* just renaming, for the output  */
   }


*-------- Standard errors
*    "predict" only gives se for the xb (linear) part, so need to get it by
*    brute force (after the probit or the logit transformation). There is nothing
*    sacred about (1.96 + 1.96), it was just conveniently available. 
*    Also: in probit & logit, se's are not symmetric around p.
quietly gen t_se = abs( t_phi - t_plow ) / (1.96 + 1.96)   


*--------- Results 
*    Results needed are the MEAN of the variables (ie mean of the col vector)
    **display "-------PROBIT: p= prob, plow= lower 95% CI, phigh= upper CI " 
    **summarize t_p t_plow t_phi

quietly egen t_p1 = mean(t_p)
quietly egen t_plow1 = mean(t_plow )
quietly egen t_phi1 = mean(t_phi )
quietly egen t_xbstd1 = mean(t_se )

*-------- Store the results
    *Store the 1st value, if this is the 1st step in the loop
    if $eval_count == 1  {
        global eval_pred1 = t_p1[1]
        *initialize value for previous run
        global eval_predLast = t_p1[1]
        global eval_change  = 0 
        global eval_change1 = 0 
        global eval_ratio = 1 
    }
    
    *Store the current value
    global eval_pred2 = t_p1[1]
    
    *Change since previous call; update for next iteration
    if $eval_pred2 != $eval_predLast {
        global eval_change1  = $eval_pred2 - $eval_predLast 
        global eval_predLast = $eval_pred2 
    } 
    
    *Change since very beginning (ie 1st call)
    global eval_change = $eval_pred2 - $eval_pred1 

    *Ratio since very beginning (ie 1st call)
    global eval_ratio = $eval_pred2 / $eval_pred1 


*-------- globals, to return results to the main routine
global eval_pred = t_p1 
global eval_se = t_xbstd1 
global eval_ciLow = t_plow1
global eval_ciHi = t_phi1
*already defined:  $eval_change1 $eval_change $eval_ratio 


drop t_*                 /* drop all temporary variables */
end 
/* ||||||||||||||||||||||||||||||||||||||||||||| */



*Subroutine fix_dummy 
 
/* ||||||||||||||||||||||||||||||||||||||||||||| */
/*Program fix_dummy    JK 22 Dec 2008 
   USES: nothing
   USED BY: fix_val (in turn used for Eval)
   Fixes sets of dummy variables to zero so Eval gets right predicted values
*/

    capture program drop fix_dummy   /*Drop previous version, so can replace */
program define fix_dummy 
     args dumVars 
    *fix_dummy  "prot cath jew none other"  -- exhaustive & exclusive 0/1 
    *            1=dumVars
    *this is a local macros; you need the wierd single quotes to use it

version 9      /* Just for drill*/
         *display "fix_dummy called"
         *display "DEBUGGING echos:"
         *display " `1' " 
         *display "Entry 1:   `dumVars' "

   capture drop t_*                 /*get rid of any temporary vars */


*#1 ----test whether Eval has done anything to the dummys
*       t_Test is 1 col per row, summing the dummies
*       For the logic see: DummyLogic_2.xls. Correct dummys will have
*       exactly one entry per row. If Eval has changed things, there
*       more (or fewer) in some rows.
quietly egen t_Test = rowtotal( `dumVars')
quietly summarize t_Test
local isOK= r(min)==1 & r(sd)==0
    *display "isOK for this set of dummies: `isOK' "

*Echo error check if this is the 1st call to Eval
    if $eval_count == 0 & `isOK' == 1 {
    *display "Note: eval_count=0"
    display "      These dummies are OK: `dumVars' "
    }
    else if $eval_count == 0 & `isOK' != 1 {
    *display "Note: eval_count=0"
    display "      SORRY: these dummies do NOT seem right: `dumVars' " 
    display "             help eval4 -- says how dummies must be set up." 
    }



*#2 ------------ If Eval HAS changed them, then recode all except 1s to 0
if `isOK' == 0 {
      *display "fix_dummy has revised these dummys: `dumVars' "
    foreach var of varlist `dumVars' {
        quietly summarize `var'
        *display r(mean)
        quietly replace `var'= 0 if r(mean)< 1
     }
     *summarize `dumVars'
     }

else {
    *display "fix_dummy did not change these: `dumVars' (but Eval may have)"
    }

drop t_*          /*delete temporary variables */
end
/* ||||||||||||||||||||||||||||||||||||||||||||| */


*fix_vars 
*Just a dummy, so Eval4 can run even if you have no interactions, etc
*This is overwritten by your custom version, if you have one. So harmless.


   capture program drop fix_vars      /*Drop previous version, so can replace */
program define fix_vars 

/* compute interactions & quadratics: "quietly replace" */
quietly display " "


end 



*Clear screen
capture program drop cls   /*Drop previous version, if any */
program define cls
 qui query
 qui loc lines = c(pagesize)
 if c(more) == "on" {
    qui set more off
    display _newline(`lines')
    qui set more on
 }
 else {
    display _newline(`lines')
 }
 end
