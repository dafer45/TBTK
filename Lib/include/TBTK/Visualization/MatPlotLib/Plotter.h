/* Copyright 2019 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @package TBTKcalc
 *  @file Plotter.h
 *  @brief Plots data.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_VISUALIZATION_MAT_PLOT_LIB_PLOTTER
#define COM_DAFER45_TBTK_VISUALIZATION_MAT_PLOT_LIB_PLOTTER

#include "TBTK/AnnotatedArray.h"
#include "TBTK/Array.h"
#include "TBTK/Property/Density.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/EigenValues.h"
#include "TBTK/Property/LDOS.h"
#include "TBTK/Property/Magnetization.h"
#include "TBTK/Property/SpinPolarizedLDOS.h"
#include "TBTK/Property/WaveFunctions.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"
#include "TBTK/Visualization/MatPlotLib/Argument.h"
#include "TBTK/Visualization/MatPlotLib/ContourfParameters.h"
#include "TBTK/Visualization/MatPlotLib/matplotlibcpp.h"
#include "TBTK/Visualization/MatPlotLib/PlotParameters.h"
#include "TBTK/Visualization/MatPlotLib/PlotSurfaceParameters.h"

#include <string>
#include <tuple>
#include <vector>


namespace TBTK{
namespace Visualization{
namespace MatPlotLib{

/** @brief Plots data.
 *
 *  The Plotter can plot @link Property::AbstractProperty Properties@endlink,
 *  @link Array Arrays@endlink, @link AnnotatedArray AnnotatedArrays@endlink,
 *  and other types that are automatically convertable to these types.
 *
 *  # Matplotlib as backend
 *  The plotter uses matplotlib as backend and therefore requires python with
 *  matplotlib to be installed to be possible to use. While the Plotter does
 *  not give direct access to matplotlib, it does allow for a subset of plot
 *  customization to be done through plot arguments. Each plot function takes
 *  an optional list of key-value pairs that will be passed on to the
 *  corresponding matplotlib function. For example, it is possible to set the
 *  line width and color as follows.
 *  ```cpp
 *    plotter.plot(data, {{"linewidth", "2"}, {"color", "red"}});
 *  ```
 *
 *  It is important to note that the Plotter switches between different
 *  matplotlib routines to plot different types of data. The possible key-value
 *  pairs depends on which routine is used by the plotter. By default, the
 *  Plotter uses the function matplotlib.pyplot.plot for 1D data and
 *  matplotlib.pyplot.contourf for 2D data.
 *
 *  # 2D plots
 *  By default 2D data is plotted using matplotlib.pyplot.contourf, but it is
 *  also possible to use mpl_toolkit.mplot3d.axes3d.Axes3D.plot_surface. The
 *  function that is used can be set using
 *  ```cpp
 *    plotter.setPlotMethod3D("plot_surface");
 *  ```
 *  or
 *  ```cpp
 *    plotter.setPlotMethod3D("contourf");
 *  ```
 *
 *  ## Number of contours for contourf
 *  When using *contourf*, the number of contour levels can be set using
 *  ```cpp
 *    plotter.setNumContours(20);
 *  ```
 *
 *  ## Rotation for plot_surface
 *  When using *plot_surface*, the rotation can be set using
 *  ```cpp
 *    plotter.setRotation(elevation, azimuthal);
 *  ```
 *  where *elevation* and *azimuthal* have the type int.
 *
 *  # Title and labels
 *  To set the title and labels, use
 *  ```cpp
 *    plotter.setTitle("My title");
 *    plotter.setLabelX("z-axis");
 *    plotter.setLabelY("y-axis");
 *    plotter.setLabelZ("z-axis");
 *  ```
 *
 *  # Modifying axes
 *  ## Bounds
 *  By default axes are rescaled to fit the data. The bounds for a given axis
 *  can be changed by calling
 *  ```cpp
 *    plotter.setBoundsX(minX, minY);
 *    plotter.setBoundsY(minY, minY);
 *  ```
 *  or
 *  ```cpp
 *    plotter.setBounds(minX, maxX, minY, maxY);
 *  ```
 *
 *  ## Ticks
 *  By default the ticks run from 0 to N-1, where N is the number of data
 *  points for the given axis. If the data contans additional information, that
 *  allows the tick values to be modified automatically, the Plotter will do
 *  so.
 *
 *  For example, @link Property::EnergyResolvedProperty
 *  EnergyResolvedProperties@endlink such as the @link Property::DOS
 *  DOS@endlink will have their ticks running between the energy range's lower
 *  and upper bound. Similarly, a Property with an Index structure such as {x,
 *  y, z}, where x runs from minX to maxX, will have its tick values for the
 *  x-axis run from minX to maxX.
 *
 *  It is possible to override this behavior as follows.
 *  ```cpp
 *    plotter.setAxes({
 *      {0, {lowerBound, upperBound}},
 *      {1, {tick0, tick1, tick2}}
 *    });
 *  ```
 *  Here the first line says that the tick values for the first axis should be
 *  replaced by uniformly spaced tick values ranging from lowerBound to
 *  upperBound. The second line says that the tick values for the second axis
 *  should be replaced by tick0, tick1, and tick2. When using the second
 *  format, it is important that the number of supplied tick values is the same
 *  as the size of the range of the data along that axis. The list of axes
 *  supplied to *setAxes* does not need to be complete and the default behavior
 *  will be applied to all axes that are not in the list.
 *
 *  # Properties
 *  @link Property::AbstractProperty Properties@endlink can have one of three
 *  different formats:
 *  - IndexDescriptor::Format::None
 *  - IndexDescriptor::Format::Ranges
 *  - IndexDescriptor::Format::Custom
 *
 *  The syntax for plotting these differs slightly.
 *
 *  ## None and Ranges
 *  @link Property::AbstractProperty Properties@endlink on the None and Ranges
 *  can be plotted using the syntax
 *  ```cpp
 *    plotter.plot(property, optionalKeyValuePairs);
 *  ```
 *  where *property* is a @link Property::AbstractProperty Property@endlink and
 *  *optionalKeyValuePairs* is an optional list of key-value pairs as described
 *  in the matplotlib section above.
 *
 *  ## Custom
 *  @link Property::AbstractProperty Properties@endlink on the Custom format
 *  does not have an explicit structural layout. The Plotter therefore need an
 *  additional pattern Index to determine how to plot the data. If, for
 *  example, the @link Property::AbstractProperty Property@endlink has the
 *  Index structure {x, y, z}, it can be plotted for the y=5 plane using
 *  ```cpp
 *    plotter.plot({_a_, 5, _a_}, property, optionalKeyValuePairs);
 *  ```
 *  The number of wildcard falgs *_a_* determines the dimensionallity of the
 *  output (plus an additional dimension if the @link
 *  Property::AbstractProperty Property@endlink has a block structure. All data
 *  points satisfying the given pattern will be organized into a grid like
 *  structure, and for possible missing data points in this grid, the value
 *  will be assumed to be zero.
 *
 *  # Example
 *  \snippet Visualization/MatPlotLib/Plotter.cpp Plotter
 *  ## Output
 *  \image html output/Visualization/MatPlotLib/Plotter/figures/VisualizationMatPlotLibPlotterArray1D.png
 *  \image html output/Visualization/MatPlotLib/Plotter/figures/VisualizationMatPlotLibPlotterDefaultLineStyles.png
 *  \image html output/Visualization/MatPlotLib/Plotter/figures/VisualizationMatPlotLibPlotterContourf.png
 *  \image html output/Visualization/MatPlotLib/Plotter/figures/VisualizationMatPlotLibPlotterPlotSurface.png
 *  \image html output/Visualization/MatPlotLib/Plotter/figures/VisualizationMatPlotLibPlotterCustomAxes.png
 *  \image html output/Visualization/MatPlotLib/Plotter/figures/VisualizationMatPlotLibPlotterFullDensity.png
 *  \image html output/Visualization/MatPlotLib/Plotter/figures/VisualizationMatPlotLibPlotterDensityCut.png */
class Plotter{
public:
	/** Default constructor. */
	Plotter();

	/** Set the canvas size.
	 *
	 *  @param width The width of the canvas.
	 *  @param height the height of the canvas. */
	void setSize(unsigned int width, unsigned int height);

	/** Set bounds for the x-axis.
	 *
	 *  @param minX The minimum value for the x-axis.
	 *  @param maxX The maximum value for the x-axis. */
	void setBoundsX(double minX, double maxX);

	/** Set bounds for the y-axis.
	 *
	 *  @param minY The minimum value for the y-axis.
	 *  @param maxY The maximum value for the y-axis. */
	void setBoundsY(double minY, double maxY);

	/** Set bounds for the x- and y-axes.
	 *
	 *  @param minX The minimum value for the x-axis.
	 *  @param maxX The maximum value for the x-axis.
	 *  @param minY The minimum value for the y-axis.
	 *  @param maxY The maximum value for the y-axis. */
	void setBounds(double minX, double maxX, double minY, double maxY);

	/** Set auto scale. */
//	void setAutoScaleX(bool autoScaleX);

	/** Set auto scale. */
//	void setAutoScaleY(bool autoScaleY);

	/** Set auto scale. */
//	void setAutoScale(bool autoScale);

	/** Set axes values. These axes will override the default axes values.
	 *  A call can take the following form.
	 *  ```cpp
	 *  plotter.setAxes({
	 *    {axisID0, {lowerBound, upperBound}},
	 *    {axisID1, {tick0, tick1, tick2}}
	 *  });
	 *  ```
	 *  Each line corresponds to a separate axis, where the axisID
	 *  specifies the axis for which to override the default behavior.
	 *
	 *  If the length of the list corresponding to a given axisID is equal
	 *  to two, the values will be interpreted as the lower and upper
	 *  bounds for the plot. If a subsequntly plotted data has N entries
	 *  for the given axis, element 0 will correspond to the tick value
	 *  lowerBound, while element N-1 will correspond to upperBound.
	 *
	 *  If the length of the list corresponding to a given axisID is
	 *  different from two, the values will be interpreted as tick values.
	 *  The number of elements in the subsequently plotted data must have
	 *  the same number of elements for the given axis.
	 *
	 *  @param axes List of axes to override the default axes with. */
	void setAxes(
		const std::vector<
			std::pair<unsigned int, std::vector<double>>
		> &axes
	);

	/** Set the plot title.
	 *
	 *  @param title The title of the plot.
	 *  @param overwrite If set to false, the title will only be set if it
	 *  has not already been set. */
	void setTitle(const std::string &title, bool overwrite = true);

	/** Set the label for the x-axis.
	 *
	 *  @param labelX The label for the x-axis.
	 *  @param overwrite If set to false, the label will only be set if it
	 *  has not already been set. */
	void setLabelX(const std::string &labelX, bool overwrite = true);

	/** Set the label for the y-axis.
	 *
	 *  @param labelY The label for the y-axis.
	 *  @param overwrite If set to false, the label will only be set if it
	 *  has not already been set. */
	void setLabelY(const std::string &labelY, bool overwrite = true);

	/** Set the label for the z-axis.
	 *
	 *  @param labelZ The label for the z-axis.
	 *  @param overwrite If set to false, the label will only be set if it
	 *  has not already been set. */
	void setLabelZ(const std::string &labelZ, bool overwrite = true);

	/** Plot point. */
//	void plot(double x, double y, const std::string &arguments);

	/** Plot density on the Property::IndexDescriptor::Format::Ranges
	 *  format.
	 *
	 *  @param density The Property::Density to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		const Property::Density &density,
		const Argument &argument = ""
	);

	/** Plot density on the the Property::IndexDescriptor::Format::Custom
	 *  format.
	 *
	 *  @param pattern An Index pattern that will be used to extract data
	 *  from the density. For example, if the Index structure of the data
	 *  contained in the density is {x, y, z}, the pattern {_a_, 5, _a_}
	 *  will result in a plot of the density in the y=5 plane.
	 *
	 *  @param density The Property::Density to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		const Index &pattern,
		const Property::Density &density,
		const Argument &argument = ""
	);

	/** Plot density of states (DOS).
	 *
	 *  @param dos The Property::DOS to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		const Property::DOS &dos,
		const Argument &argument = ""
	);

	/** Plot eigenvalues.
	 *
	 *  @param dos The Property::EigenValues to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		const Property::EigenValues &eigenValues,
		const Argument &argument = "black"
	);

	/** Plot local density of states (LDOS) on the
	 *  IndexDescriptor::Format::Ranges format.
	 *
	 *  @param ldos The Property::LDOS to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(const Property::LDOS &ldos, const Argument &argument = "");

	/** Plot local density of states (LDOS) on the
	 *  IndexDescriptor::Format::Custom format.
	 *
	 *  @param pattern An Index pattern that will be used to extract data
	 *  from the LDOS. For example, if the Index structure of the data
	 *  contained in the LDOS is {x, y, z}, the pattern {5, _a_, 10}
	 *  will result in a plot of the LDOS along the line (x, z) = (5, 10).
	 *
	 *  @param ldos The Property::LDOS to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		const Index &pattern,
		const Property::LDOS &ldos,
		const Argument &argument = ""
	);

	/** Plot magnetization on the IndexDescriptor::Format::Ranges format.
	 *
	 *  @param direction The quantization axis to use.
	 *  @param magnetization The Property::Magnetization to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		const Vector3d &direction,
		const Property::Magnetization &magnetization,
		const Argument &argument = ""
	);

	/** Plot magnetization on the IndexDescriptor::Format::Ranges format.
	 *
	 *  @param pattern An Index pattern that will be used to extract data
	 *  from the Magnetization. For example, if the Index structure of the
	 *  data contained in the Magnetization is {x, y, z}, the pattern
	 *  {5, _a_, 10} will result in a plot of the Magnetization along the
	 *  line (x, z) = (5, 10).
	 *
	 *  @param direction The quantization axis to use.
	 *  @param magnetization The Property::Magnetization to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		const Index &pattern,
		const Vector3d &direction,
		const Property::Magnetization &magnetization,
		const Argument &argument = ""
	);

	/** Plot spin-polarized local density of states (LDOS) on the
	 *  IndexDescriptor::Format::Ranges format.
	 *
	 *  @param direction The quantization axis to use.
	 *  @param spinPolarizedLDOS The Property::SpinPolarizedLDOS to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		const Vector3d &direction,
		const Property::SpinPolarizedLDOS &spinPolarizedLDOS,
		const Argument &argument = ""
	);

	/** Plot spin-polarized local density of states (LDOS) on the
	 *  IndexDescriptor::Format::Ranges format.
	 *
	 *  @param pattern An Index pattern that will be used to extract data
	 *  from the SpinPolarizedLDOS. For example, if the Index structure of
	 *  the data contained in the SpinPolarizedLDOS is {x, y, z}, the
	 *  pattern {5, _a_, 10} will result in a plot of the SpinPolarizedLDOS
	 *  along the line (x, z) = (5, 10).
	 *
	 *  @param direction The quantization axis to use.
	 *  @param spinPolarizedLDOS The Property::SpinPolarizedLDOS to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		const Index &pattern,
		const Vector3d &direction,
		const Property::SpinPolarizedLDOS &spinPolarizedLDOS,
		const Argument &argument = ""
	);

	/** Plot wave function on the IndexDescriptor::Format::Ranges format.
	 *
	 *  @param state The state number to plot the wave function for.
	 *  @param waveFunctions The Property::WaveFunctions to plot from.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		unsigned int state,
		const Property::WaveFunctions &waveFunctions,
		const Argument &argument = ""
	);

	/** Plot wave function on the IndexDescriptor::Format::Ranges format.
	 *
	 *  @param pattern An Index pattern that will be used to extract data
	 *  from the WaveFunctions. For example, if the Index structure of the
	 *  data contained in the WaveFunctions is {x, y, z}, the pattern
	 *  {5, _a_, 10} will result in a plot of the WaveFunctions along the
	 *  line (x, z) = (5, 10).
	 *
	 *  @param state The state number to plot the wave function for.
	 *  @param waveFunctions The Property::WaveFunctions to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		const Index &pattern,
		unsigned int state,
		const Property::WaveFunctions &waveFunctions,
		const Argument &argument = ""
	);

	/** Plot arbitrary data in stored in an AnnotatedArray.
	 *
	 *  @param data The data to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		const AnnotatedArray<double, double> &data,
		const Argument &argument = ""
	);

	/** Plot arbitrary data stored in an AnnotatedArray.
	 *
	 *  @param data The data to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		const AnnotatedArray<double, Subindex> &data,
		const Argument &argument = ""
	);

	/** Plot arbitrary data stored in an Array.
	 *
	 *  @param data The data to plot.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(const Array<double> &data, const Argument &argument = "");

	/** Plot arbitrary data stored in @link Array Arrays@endlink.
	 *
	 *  @param x The data for the x-axis.
	 *  @param y The data for the y-axis.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		Array<double> x,
		const Array<double> &y,
		const Argument &argument = ""
	);

	/** Plot arbitrary data stored in an Array.
	 *
	 *  @param x The data for the x-axis.
	 *  @param y The data for the y-axis.
	 *  @param z The data for the z-axis.
	 *  @param argument A list of arguments to pass to the underlying
	 *  matplotlib function. Can either be a single string value or a list
	 *  such as {{"linewidth", "2"}, {"color", "red"}}. */
	void plot(
		Array<double> x,
		Array<double> y,
		const Array<double> &z,
		const Argument &argument = ""
	);

	/** Plot data with color coded intensity. */
/*	void plot(
		const std::vector<std::vector<double>> &data,
		const std::vector<std::vector<double>> &intensity,
		const std::string &arguments
	);*/

	/** Plot data with color coded intensity. */
/*	void plot(
		const Array<double> &data,
		const Array<double> &intensity,
		const std::string &arguments
	);*/

	/** Set the plot method to use for 3D data.
	 *
	 *  @param plotMethod3D The name of the matplotlib function to use for
	 *  3D data. The currently supported values are "contourf" and
	 *  "plot_surface". */
	void setPlotMethod3D(const std::string &plotMethod3D);

	/** Set rotation angels for 3D plots.
	 *
	 *  @param elevation The elevation angle.
	 *  @param azimuthal The azimuthal angle.
	 *  @param overwrite If set to false, the angles will only be set if
	 *  they have not already been set. */
	void setRotation(int elevation, int azimuthal, bool overwrite = true);

	/** Set the number of contours to use when plotting contour plots.
	 *
	 *  @param numContours The number of contour levels to use when
	 *  plotting contour plots. */
	void setNumContours(unsigned int numContours);

	/** Set whether ot not data is plotted on top of old data. */
//	void setHold(bool hold);

	/** Clear the plot and all configuration data. */
	void clear();

	/** Show the plot using matplotlibs GUI. */
	void show() const;

	/** Save the canvas to file.
	 *
	 *  @param filename The file to save the canvas to. */
	void save(const std::string &filename) const;

	/** @name Ambiguity resolution
	 *  @{
	 *  These functions resolves otherwise ambiguous function calls.
	 *  */
	void plot(
		const Array<double> &data,
		const std::initializer_list<
			std::pair<std::string, std::string>
		> &argument
	){
		plot(data, Argument(argument));
	}

	void plot(
		const std::initializer_list<double> &data,
		const Argument &argument = ""
	){
		plot(std::vector<double>(data), argument);
	}

	void plot(
		const std::initializer_list<double> &x,
		const Array<double> &y,
		const Argument &argument = ""
	){
		plot(std::vector<double>(x), y, argument);
	}

	void plot(
		const Array<double> &x,
		const std::initializer_list<double> &y,
		const Argument &argument = ""
	){
		plot(x, std::vector<double>(y), argument);
	}

	void plot(
		const std::initializer_list<double> &x,
		const std::initializer_list<double> &y,
		const Argument &argument = ""
	){
		plot(
			std::vector<double>(x),
			std::vector<double>(y),
			argument
		);
	}
	/** @} */
private:
	/** Enum class for keeping track of the current type of plot. */
	enum class CurrentPlotType{None, Plot1D, PlotSurface, Contourf};

	/** Current plot type. */
	CurrentPlotType currentPlotType;

	/** Enum class for keeping track of the plot method to use for 3D data.
	 */
	enum class PlotMethod3D{PlotSurface, Contourf};

	/** The plot method to use for 3D data. */
	PlotMethod3D plotMethod3D;

	/** Parameters for plots using plot. */
	PlotParameters plotParameters;

	/** Parameters for plots using plot_surface. */
	PlotSurfaceParameters plotSurfaceParameters;

	/** Parameters for plots using contourf. */
	ContourfParameters contourfParameters;

	/** Axes to use instead of the default axes. */
	std::vector<std::pair<unsigned int, std::vector<double>>> axes;

	/** Number of currently ploted lines. */
	unsigned int numLines;

	/** The number of contours to use when plotting contour plots. */
	unsigned int numContours;

	/** Plot data. */
	void plot1D(
		const std::vector<double> &y,
		const Argument &argument = ""
	);

	/** Plot data. */
	void plot1D(
		const std::vector<double> &x,
		const std::vector<double> &y,
		const Argument &argument = ""
	);

	/** Plot 2D data. */
	void plot2D(
		const std::vector<std::vector<double>> &z,
		const Argument &argument = ""
	);

	/** Plot 2D data. */
	void plot2D(
		const std::vector<std::vector<double>> &x,
		const std::vector<std::vector<double>> &y,
		const std::vector<std::vector<double>> &z,
		const Argument &argument = ""
	);

	/** Convert an AnnotatedArray with Subindex axes to an AnnotatedArray
	 *  with custom double valued axes. By default, axis values are simply
	 *  converted from Subindex to double. A list of values such as
	 *  {axisID, {lowerBound, upperBound}} can be supplied to change the
	 *  values for individual axes. Here axisID is the axis to modify,
	 *  while lowerBound and upperBound are the new bounds for the
	 *  corresponding axis. */
	AnnotatedArray<double, double> convertAxes(
		const AnnotatedArray<double, Subindex> &annotatedArray,
		const std::initializer_list<
			std::pair<unsigned int, std::pair<double, double>>
		> &axisReplacement = {}
	);

	/** Returns an Array containing the axis data that should be used when
	 *  plotting. If non-default axes have been set for the given axisID,
	 *  an axis will be returned that reflects the non-default axis.
	 *  Otherwise, axis will be returned. */
	Array<double> getNonDefaultAxis(
		const Array<double> &axis,
		unsigned int axisID
	) const;

	/** Converts a SpinMatrix valued AnnotatedArray to a double valued
	 *  AnnotatedArray by projecting the spin matrix on a give quantization
	 *  axis. */
	AnnotatedArray<double, Subindex> convertSpinMatrixToDouble(
		const AnnotatedArray<SpinMatrix, Subindex> &annotatedArray,
		const Vector3d &direction
	) const;

	/** Convert a color to hexdecimal. */
	std::string colorToHex(const Array<double> &color) const;

	/** Convert double to hexadecimal value. */
	std::string doubleToHex(double value) const;
};

inline Plotter::Plotter(){
	currentPlotType = CurrentPlotType::None;
	plotMethod3D = PlotMethod3D::Contourf;
	numLines = 0;
	numContours = 8;
	clear();
}

inline void Plotter::setBoundsX(
	double minX,
	double maxX
){
	TBTKAssert(
		minX < maxX,
		"Plotter::setBoundsX()",
		"minX has to be smaller than maxX",
		""
	);
	plotParameters.setBoundsX(minX, maxX);
	contourfParameters.setBoundsX(minX, maxX);
	plotSurfaceParameters.setBoundsX(minX, maxX);
}

inline void Plotter::setBoundsY(
	double minY,
	double maxY
){
	TBTKAssert(
		minY < maxY,
		"Plotter::setBoundsY()",
		"minY has to be smaller than maxY",
		""
	);
	plotParameters.setBoundsY(minY, maxY);
	contourfParameters.setBoundsY(minY, maxY);
	plotSurfaceParameters.setBoundsY(minY, maxY);
}

inline void Plotter::setBounds(
	double minX,
	double maxX,
	double minY,
	double maxY
){
	setBoundsX(minX, maxX);
	setBoundsY(minY, maxY);
}

/*inline void Plotter::setAutoScaleX(bool autoScaleX){
	this->autoScaleX = autoScaleX;
}

inline void Plotter::setAutoScaleY(bool autoScaleY){
	this->autoScaleY = autoScaleY;
}

inline void Plotter::setAutoScale(bool autoScale){
	setAutoScaleX(autoScale);
	setAutoScaleY(autoScale);
}*/

inline void Plotter::setAxes(
	const std::vector<
		std::pair<unsigned int, std::vector<double>>
	> &axes
){
	this->axes = axes;
}

inline void Plotter::setTitle(const std::string &title, bool overwrite){
	plotParameters.setTitle(title, overwrite);
	plotSurfaceParameters.setTitle(title, overwrite);
	contourfParameters.setTitle(title, overwrite);
	matplotlibcpp::title(title);
}

inline void Plotter::setLabelX(const std::string &labelX, bool overwrite){
	plotParameters.setLabelX(labelX, overwrite);
	plotSurfaceParameters.setLabelX(labelX, overwrite);
	contourfParameters.setLabelX(labelX, overwrite);
	matplotlibcpp::xlabel(labelX);
}

inline void Plotter::setLabelY(const std::string &labelY, bool overwrite){
	plotParameters.setLabelY(labelY, overwrite);
	plotSurfaceParameters.setLabelY(labelY, overwrite);
	contourfParameters.setLabelY(labelY, overwrite);
	matplotlibcpp::ylabel(labelY);
}

inline void Plotter::setLabelZ(const std::string &labelZ, bool overwrite){
	plotSurfaceParameters.setLabelZ(labelZ, overwrite);
}

inline void Plotter::setPlotMethod3D(const std::string &plotMethod3D){
	if(plotMethod3D.compare("plot_surface") == 0){
		this->plotMethod3D = PlotMethod3D::PlotSurface;
	}
	else if(plotMethod3D.compare("contourf") == 0){
		this->plotMethod3D = PlotMethod3D::Contourf;
	}
	else{
		TBTKExit(
			"Plotter::setPlotMethod3D()",
			"Unknown plot method.",
			"Must be 'plot_surface' or 'contourf'."
		);
	}
}

inline void Plotter::setRotation(int elevation, int azimuthal, bool overwrite){
	plotSurfaceParameters.setRotation(elevation, azimuthal, overwrite);
	switch(currentPlotType){
	case CurrentPlotType::PlotSurface:
		plotSurfaceParameters.flush();
		break;
	default:
		break;
	}
}

inline void Plotter::setNumContours(unsigned int numContours){
	this->numContours = numContours;
}

/*inline void Plotter::setHold(bool hold){
	this->hold = hold;
}*/

inline void Plotter::clear(){
	plotParameters.clear();
	plotSurfaceParameters.clear();
	contourfParameters.clear();
	axes.clear();
	numLines = 0;
	numContours = 8;
	matplotlibcpp::clf();
}

inline void Plotter::show() const{
	matplotlibcpp::show();
}

inline void Plotter::save(const std::string &filename) const{
	matplotlibcpp::save(filename);
}

};	//End namespace MatPlotLib
};	//End namespace Visualization
};	//End namespace TBTK

#endif
