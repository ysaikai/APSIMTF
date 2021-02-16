namespace Models
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.IO;
    using Models.Core;
    using Models.Core.Run;
    using Models.Interfaces;
    using Newtonsoft.Json;
    using Newtonsoft.Json.Linq;
    using APSIM.Shared.Utilities;

    /// <summary>Class to hold a static main entry point.</summary>
    public class Program
    {
        /// <summary>policy network</summary>
        public static PolicyNet policyNet;
        /// <summary> A set of irrigation amounts </summary>
        public static float[] amounts = { 0f, 20f, 40f, 60f, 80f };
        /// <summary></summary>
        public static float yield;
        /// <summary></summary>
        public static List<float> days;
        /// <summary></summary>
        public static List<float> lais;
        /// <summary></summary>
        public static List<float> esws;
        /// <summary></summary>
        public static List<float> cuIrrigs;
        /// <summary></summary>
        public static List<int> actions;
        /// <summary></summary>
        public static List<float[]> probs;

        /// <summary></summary>
        public static void Main(string[] args)
        {
            // Parameters
            int n_episodes;
            int rndSeed;
            try // console inputs
            {
                n_episodes = Int32.Parse(args[0]);
                rndSeed = Int32.Parse(args[1]);
            }
            catch // default
            {
                n_episodes = 1000;
                rndSeed = (int) DateTime.Now.Ticks & 0x0000FFFF;
            }
            int order = 50; // Order of MA
            float lr = 1e-3f;
            int[] n_hidden = { 128, 256, 512, 256, 128 };
            string location = "Dalby";
            // Dalby
            int yearStart = 1900;
            int yearEnd = 2000;
            var yearsExcluded = new List<int> { 1906, 1908, 1911, 1915, 1917, 1918, 1924, 1929,
                                                1933, 1937, 1941, 1943, 1944, 1946, 1954, 1960,
                                                1962, 1971, 1972, 1984, 1991, 1994 };

            // Variables
            float ma = 0f; // moving average
            float maMax = 0f;
            int episodeMax = 0;
            Random rand = new Random(rndSeed);
            int year;
            policyNet = new PolicyNet(n_hidden, rand);
            PolicyNet policyNetMax = new PolicyNet(n_hidden, rand);

            // Relative path
            string path = System.AppDomain.CurrentDomain.BaseDirectory;
            //path = System.IO.Directory.GetParent(path).ToString();
            path = System.IO.Path.GetDirectoryName(path);
            path = System.IO.Path.GetDirectoryName(path);

            // APSIM file
            string fileNameAPSIM = path + "\\Wheat.apsimx";

            // Welcome message
            string msg = "\n";
            msg += "~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~\n";
            msg += "{0} episodes; MA({1}); seed={2}; H1-5={3}-{4}-{5}-{6}-{7}\n";
            msg += "~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~";
            Console.WriteLine(msg, n_episodes, order, rndSeed,
                              n_hidden[0], n_hidden[1], n_hidden[2], n_hidden[3], n_hidden[4]);

            // Run
            var stopWatch = new System.Diagnostics.Stopwatch();
            for (int i = 0; i < n_episodes; i++)
            {
                stopWatch.Start();
                Initialize();
                do
                    year = rand.Next(yearStart, yearEnd + 1);
                while (yearsExcluded.Contains(year));
                ProcessApsimFile(fileNameAPSIM, year, location);
                var runner = new Runner(fileNameAPSIM);
                runner.Run();

                // Update
                ma = (ma * (order - 1) + yield) / order;
                if (ma > maMax)
                {
                    maMax = ma;
                    episodeMax = i;
                    policyNetMax = policyNet.DeepCopy(); // Save the best policy
                }
                policyNet.Update(lr);

                // Progress
                if (i % 100 == 0)
                {
                    stopWatch.Stop();
                    TimeSpan ts = stopWatch.Elapsed;
                    Console.WriteLine("{0,5:D1}: {1,4:F0} / {2:0} (at {3}) {4:00}:{5:00}",
                                      i, ma, maMax, episodeMax, ts.Hours, ts.Minutes);
                }
            }

            // Results
            policyNet = policyNetMax; // Demonstrate the best policy
            for (year = yearStart; year < yearEnd+1; year++)
            {
                if (yearsExcluded.Contains(year)) // Skip the excluded years
                    continue;
                Initialize();
                ProcessApsimFile(fileNameAPSIM, year, location);
                var runner = new Runner(fileNameAPSIM);
                runner.Run();

                // CSV output
                if (!Directory.Exists(path + "\\results"))
                    Directory.CreateDirectory(path + "\\results");
                using (var file = new StreamWriter(path + $"\\results\\{year}.csv"))
                {
                    file.WriteLine($"{yield}");
                    file.WriteLine($"{rndSeed}");
                    file.WriteLine("");
                    file.WriteLine("Day,  LAI, ESW, cuIrrig, Action, p(0), p(1), p(2), p(3), p(4)");
                    for (int i = 0; i < days.Count; i++)
                    {
                        file.Write("{0:0}, {1,4:F2}, {2,3:F0}, {3,7:F0}, {4,6:D1}, ",
                                       days[i],
                                       lais[i],
                                       esws[i],
                                       cuIrrigs[i],
                                       actions[i]);
                        file.WriteLine("{0:0.00}, {1:0.00}, {2:0.00}, {3:0.00}, {4:0.00}",
                                       probs[i][0],
                                       probs[i][1],
                                       probs[i][2],
                                       probs[i][3],
                                       probs[i][4]);
                    }
                }
            }
        }


        static void Initialize()
        {
            days = new List<float>();
            lais = new List<float>();
            esws = new List<float>();
            cuIrrigs = new List<float>();
            actions = new List<int>();
            probs = new List<float[]>();
        }


        // Process .apsimx file as JSON file (https://stackoverflow.com/a/56027969)
        static void ProcessApsimFile(string fileNameAPSIM, int year, string location)
        {
            string jsonString = File.ReadAllText(fileNameAPSIM);
            JObject jObject = JsonConvert.DeserializeObject(jsonString) as JObject;
            JToken jtStart = jObject.SelectToken("Children[0].Children[0].Start");
            jtStart.Replace(year.ToString() + "-01-01T00:00:00");
            JToken jtEnd = jObject.SelectToken("Children[0].Children[0].End");
            jtEnd.Replace(year.ToString() + "-12-31T00:00:00");
            JToken jtLoc = jObject.SelectToken("Children[0].Children[2].FileName");
            jtLoc.Replace(".\\" + location + ".met");
            string updatedJsonString = jObject.ToString();
            File.WriteAllText(fileNameAPSIM, updatedJsonString);
        }
    }
}
