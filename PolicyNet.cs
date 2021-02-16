using NumSharp;
using System;
using System.Data;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras.Optimizers;
using static Tensorflow.Binding;

namespace Models
{
    /// <summary></summary>
    public class PolicyNet
    {
        int n_features = 4;
        int n_classes = 5;

        // Initial factors for standardization (from demo.apsim)
        NDArray means = np.array(new float[] { 217f, 1.27f, 389f, 180f });
        NDArray stds = np.array(new float[] { 55.4f, 1.63f, 37.5f, 87.3f });

        IVariableV1 h1, h2, h3, h4, h5, wout, b1, b2, b3, b4, b5, bout;
        IVariableV1[] trainable_variables;
        OptimizerV2 optimizer;
        Random rand;

        /// <summary></summary>
        public PolicyNet DeepCopy()
        {
            PolicyNet clonedNet = (PolicyNet)this.MemberwiseClone();
            clonedNet.h1 = tf.Variable(h1);
            clonedNet.h2 = tf.Variable(h2);
            clonedNet.h3 = tf.Variable(h3);
            clonedNet.h4 = tf.Variable(h4);
            clonedNet.h5 = tf.Variable(h5);
            clonedNet.wout = tf.Variable(wout);
            clonedNet.b1 = tf.Variable(b1);
            clonedNet.b2 = tf.Variable(b2);
            clonedNet.b3 = tf.Variable(b3);
            clonedNet.b4 = tf.Variable(b4);
            clonedNet.b5 = tf.Variable(b5);
            clonedNet.bout = tf.Variable(bout);
            clonedNet.means = (NDArray)means.Clone();
            clonedNet.stds = (NDArray)stds.Clone();

            return clonedNet;
        }

        /// <summary></summary>
        public PolicyNet(int[] n_hidden, Random rnd)
        {
            // TODO: Where to put this statement? Is it needed? Isn't it default in TF2?
            tf.enable_eager_execution();

            rand = rnd;
            var ini = tf.initializers.random_normal_initializer(seed: rand.Next());
            h1 = tf.Variable(ini.Apply(new InitializerArgs((n_features, n_hidden[0]))));
            h2 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[0], n_hidden[1]))));
            h3 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[1], n_hidden[2]))));
            h4 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[2], n_hidden[3]))));
            h5 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[3], n_hidden[4]))));
            wout = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[4], n_classes))));
            b1 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[0]))));
            b2 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[1]))));
            b3 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[2]))));
            b4 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[3]))));
            b5 = tf.Variable(ini.Apply(new InitializerArgs((n_hidden[4]))));
            bout = tf.Variable(ini.Apply(new InitializerArgs((n_classes))));

            trainable_variables = new IVariableV1[] { h1, h2, h3, h4, h5, wout,
                                                      b1, b2, b3, b4, b5, bout };
        }


        /// <summary></summary>
        public void Update(float learning_rate)
        {
            // Store states in NDArray and standardize
            NDArray S = np.hstack(np.array(Program.days).reshape(-1, 1),
                                  np.array(Program.lais).reshape(-1, 1),
                                  np.array(Program.esws).reshape(-1, 1),
                                  np.array(Program.cuIrrigs).reshape(-1, 1));
            NDArray meansOld = (NDArray)means.Clone();
            NDArray stdsOld = (NDArray)stds.Clone();
            for (int i = 0; i < 4; i++)
            {
                var column = S[$":,{i}"];
                // New factors are assumed to be simply average of the previous and current
                means[i] = (column.mean() + meansOld[i]) / 2; // new mean
                stds[i] = (column.std() + stdsOld[i]) / 2; // new std
                S[$":,{i}"] = (column - means[i]) / stds[i]; // scaled states
            }

            // A is used for masking, i.e. multiplying probabilities for unchosen actions by 0
            var A = tf.one_hot(np.array(Program.actions), n_classes);

            // To make use of masking by A, G has identical column entries at each row
            int n_days = Program.days.Count; // Same size in all 4 state variables
            float g = Program.yield;
            var G = np.ones((n_days, n_classes), dtype: np.float32); // initialization
            for (int i = n_days - 1; i >= 0; i--) // start from the harvest day
            {
                //int a = Program.actions[i];
                //g -= Program.amounts[a]; // cost of irrigation
                G[i] = G[i] * g; // each row (i) has the identical entry (g)
            }

            // Gradient ascent
            optimizer = tf.optimizers.SGD(learning_rate);
            using (var tape = tf.GradientTape())
            {
                var PROBS = Predict(S);
                // After component-wise multiplication by A, each row has only 1 nonzero entry
                var loss = tf.reduce_mean(-tf.math.log(PROBS) * A * G);
                var gradients = tape.gradient(loss, trainable_variables);
                optimizer.apply_gradients(zip(gradients,
                                trainable_variables.Select(x => x as ResourceVariable)));
            }
        }


        /// <summary>Called by the IrrigationPolicy script</summary>
        public Tuple<int, float[]> Action(float day, float lai, float esw, float cuIrrig)
        {
            //var x = np.array(new float[] { day, lai, esw, cuIrrig });
            var x = new NDArray(new float[] { day, lai, esw, cuIrrig });

            // Standardization
            for (int i = 0; i < 4; i++)
                x[i] = (x[i] - means[i]) / stds[i];

            float[] probs = Predict(x.reshape(1, 4)).ToArray<float>();
            //// We just want:
            //// int a = np.random.choice(probs.Count, probabilities: probs);
            //// But, for some reason, it spits out an error. Hence, manual implemntation.
            float r = (float)rand.NextDouble();
            float sum = 0f;
            for (int a = 0; a < n_classes; a++)
            {
                sum += probs[a];
                if (sum >= r)
                    return Tuple.Create(a, probs);
            }
            // This is never reached but the compiler requires.
            return Tuple.Create(n_classes - 1, probs);
        }


        /// <summary></summary>
        public Tensor Predict(Tensor x)
        {
            var layer_1 = tf.add(tf.matmul(x, h1.AsTensor()), b1.AsTensor());
            layer_1 = tf.nn.relu(layer_1);
            var layer_2 = tf.add(tf.matmul(layer_1, h2.AsTensor()), b2.AsTensor());
            layer_2 = tf.nn.relu(layer_2);
            var layer_3 = tf.add(tf.matmul(layer_2, h3.AsTensor()), b3.AsTensor());
            layer_3 = tf.nn.relu(layer_3);
            var layer_4 = tf.add(tf.matmul(layer_3, h4.AsTensor()), b4.AsTensor());
            layer_4 = tf.nn.relu(layer_4);
            var layer_5 = tf.add(tf.matmul(layer_4, h5.AsTensor()), b5.AsTensor());
            layer_5 = tf.nn.relu(layer_5);
            var out_layer = tf.matmul(layer_5, wout.AsTensor()) + bout.AsTensor();
            return tf.nn.softmax(out_layer);
        }
    }
}
