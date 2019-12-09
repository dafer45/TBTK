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

/// @cond TBTK_FULL_DOCUMENTATION
/** @package TBTKcalc
 *  @file HoppingAmplitudePanel.h
 *  @brief Panel for displaying a HoppingAmplitude.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_HOPPING_AMPLITUDE_PANEL
#define COM_DAFER45_TBTK_HOPPING_AMPLITUDE_PANEL

#include "TBTK/HoppingAmplitude.h"
#include "TBTK/IndexPanel.h"

#include <wx/wx.h>
#include <wx/sizer.h>

namespace TBTK{

class HoppingAmplitudePanel : public wxPanel{
public:
	/** Constructor. */
	HoppingAmplitudePanel(wxWindow *parent);

	/** Destructor. */
	~HoppingAmplitudePanel();

	/** Set HoppingAmplitude. */
	void setHoppingAmplitude(const HoppingAmplitude &hoppingAmplitude);

	/** Get HoppingAmplitude. */
	const HoppingAmplitude& getHoppingAmplitude() const;

	/** On paint event. */
	void onPaintEvent(wxPaintEvent &event);

	/** On size event. */
	void onSizeEvent(wxSizeEvent &event);

	/** On amplitude changed. */
	void onAmplitudeChanged(wxCommandEvent &event);

	/** On Index changed. */
	void onIndexChanged(wxCommandEvent &event);

	/** Redraw. */
//	void redraw();
protected:
	DECLARE_EVENT_TABLE();
private:
	/** The HoppingAmplitude. */
	HoppingAmplitude hoppingAmplitude;

	/** HoppingAmplitude label. */
	wxStaticText hoppingAmplitudeLabel;

	/** Text box for the amplitude value. */
	wxTextCtrl amplitudeTextBox;

	/** To Index panel. */
	IndexPanel toIndexPanel;

	/** From Index panel. */
	IndexPanel fromIndexPanel;

	static const wxWindowID AMPLITUDE_ID;
	static const wxWindowID TO_INDEX_ID;
	static const wxWindowID FROM_INDEX_ID;
};

inline const HoppingAmplitude& HoppingAmplitudePanel::getHoppingAmplitude() const{
	return hoppingAmplitude;
}

};	//End namespace TBTK

#endif
/// @endcond
