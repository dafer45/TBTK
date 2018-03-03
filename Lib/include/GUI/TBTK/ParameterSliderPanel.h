/* Copyright 2017 Kristofer Björnson
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
 *  @file ParameterSlider.h
 *  @brief Panel for selecting and displaying a parameter.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PARAMETER_SLIDER_PANEL
#define COM_DAFER45_TBTK_PARAMETER_SLIDER_PANEL

#include <wx/wx.h>
#include <wx/sizer.h>

namespace TBTK{

class ParameterSliderPanel : public wxPanel{
public:
	/** Constructor. */
	ParameterSliderPanel(
		wxWindow *parent,
		wxWindowID id,
		const std::string &parameterName,
		double lowerBound,
		double upperBound,
		unsigned int numTicks,
		double value
	);

	/** Destructor. */
	~ParameterSliderPanel();

	/** On paint event. */
	void onPaintEvent(wxPaintEvent &event);

	/** On size event. */
	void onSizeEvent(wxSizeEvent &event);

	/** On slider change. */
	void onSliderChange(wxScrollEvent &event);

	/** Get value. */
	double getValue() const;

	/** Get tick. */
	unsigned int getTick() const;
protected:
	DECLARE_EVENT_TABLE();
private:
	/** Parameter value. */
	double value;

	/** Lower bound. */
	double lowerBound;

	/** Upper bound. */
	double upperBound;

	/** Number of ticks. */
	unsigned int numTicks;

	/** Parameter label. */
	wxStaticText label;

	/** Slider for index. */
	wxSlider slider;

	/** Parameter value panel. */
	wxStaticText valuePanel;

	static const wxWindowID SLIDER_ID;
};

inline double ParameterSliderPanel::getValue() const{
	return value;
}

inline unsigned int ParameterSliderPanel::getTick() const{
	return slider.GetValue();
}

};	//End namespace TBTK

#endif
