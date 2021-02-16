// Policy is active between sowing and harvesting

using Models.Utilities;
using Models.Core;
using System;

namespace Models
{
    [Serializable]
    public class Script : Model
    {
		[Link] Irrigation irrigation;
        [Link] Clock clock;
        [Link] PMF.Plant wheat;
        [Link] WaterModel.WaterBalance waterBalance;
        
        float day;
		float lai;
		float esw;
		float cuIrrig = 0f;
		int action;
		float[] probs;
        bool irrigationOn = false;


		// This can be "StartOfDay", which precedes "DoManagement" in each day
        [EventSubscribe("Sowing")]
        private void OnSowing(object sender, EventArgs e)
		{
			irrigationOn = true;
		}


        [EventSubscribe("DoManagement")]
        private void OnDoManagement(object sender, EventArgs e)
        {
			if (irrigationOn)
			{
	        	// State
				Program.cuIrrigs.Add(cuIrrig); // Here, for one-day lag
				day = (float)clock.Today.DayOfYear;
				lai = (float)wheat.LAI;
				esw = 0f;
				Array.ForEach(waterBalance.ESW, item => esw += (float)item);
				
				// Action
 				var result = Program.policyNet.Action(day, lai, esw, cuIrrig);
 				action = result.Item1;
 				probs = result.Item2;
				irrigation.Apply(Program.amounts[action]);
			}
        }


        [EventSubscribe("EndOfDay")]
        private void OnEndOfDay(object sender, EventArgs e)
        {
			if (irrigationOn)
			{
	        	Program.days.Add(day);
	        	Program.lais.Add(lai);
	        	Program.esws.Add(esw);
	        	cuIrrig += Program.amounts[action];
	        	//Program.cuIrrigs.Add(cuIrrig); // Not here, for one-day lag
	        	Program.actions.Add(action);
	        	Program.probs.Add(probs);
			}
        }


        [EventSubscribe("Harvesting")]
        private void OnHarvesting(object sender, EventArgs e)
        {
        	irrigationOn = false;
        }
    }
}
