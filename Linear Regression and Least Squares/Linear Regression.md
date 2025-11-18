The inputs are a set of [[Feature]]s generally represented by x values.

The output is a scalar y value.

The model defines a relationship between the input [[Feature]]s and the output, generally by way of [[Weight]]s.

There is also an [[Error]] term to determine the difference between the model's prediction and the actual observed value.

A single observation in a [[Linear Regression]] model might look like this:

$$y = w_0 + w_1x_1 + w_2x_2 \hspace{3mm} ... \hspace{3mm} + w_px_p + \epsilon _1$$

This can also be expressed as $x^Tw + \epsilon$.

More commonly, you will have multiple observations from 1 to $n$:

$$y_1 = w_0 + w_1x_1 + w_2x_2 \hspace{3mm} ... \hspace{3mm} + w_px_p + \epsilon _1$$
$$y_2 = w_0 + w_1x_1 + w_2x_2 \hspace{3mm} ... \hspace{3mm} + w_px_p + \epsilon _2$$
$$\vdots$$
$$y_n = w_0 + w_1x_1 + w_2x_2 \hspace{3mm} ... \hspace{3mm} + w_px_p + \epsilon _n$$
The objective is to find a set of [[Weight]]s that minimises the [[Error]] to form a model that best fits the data. This can be achieved by using a [[Loss Function]]. The [[Loss Function]] used in [[Linear Regression]] is the [[Mean Squared Error]] which is the average of the sum of all the [[Error]]s squared.

To find the minimum of the [[Error]], we need to calculate the [[Gradient]].

$$\frac{\partial E}{\partial w} = (\frac{\partial E}{\partial w_0}, \hspace{3mm} ... \hspace{3mm} , \frac{\partial E}{\partial w_p})$$
A [[Linear Regression]] model needs to be linear in the [parameters](Weight) but not necessarily in the input [[Feature]]s. It is important to verify a model is [[Linear or Not]].
