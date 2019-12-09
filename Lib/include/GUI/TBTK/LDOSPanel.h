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
 *  @file LDOSPanel.h
 *  @brief Panel for displaying LDOS.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_LDOS_PANEL
#define COM_DAFER45_TBTK_LDOS_PANEL

#include "TBTK/Index.h"
#include "TBTK/ImagePanel.h"
#include "TBTK/IndexPanel.h"
#include "TBTK/Property/LDOS.h"

#include <wx/wx.h>
#include <wx/sizer.h>

namespace TBTK{

class LDOSPanel : public wxPanel{
public:
	/** Constructor. */
	LDOSPanel(wxWindow *parent);

	/** Destructor. */
	~LDOSPanel();

	/** Set LDOS. */
	void setLDOS(const Property::LDOS &ldos);

	/** On paint event. */
	void onPaintEvent(wxPaintEvent &event);

	/** On size event. */
	void onSizeEvent(wxSizeEvent &event);

	/** On change. */
	void onChange(wxCommandEvent &event);
protected:
	DECLARE_EVENT_TABLE();
private:
	/** Pointer to the property. */
	Property::LDOS *ldos;

	/** Panel for the actual DOS. */
	ImagePanel *resultPanel;

	class ControlPanel : public wxPanel{
	public:
		/** Constructor. */
		ControlPanel(wxWindow *parent);

		/** Destructor. */
		~ControlPanel();

		/** Get gaussian smoothing. */
		double getGaussianSmoothing() const;

		/** Get min. */
		double getMin() const;

		/** Get max. */
		double getMax() const;

		/** Get Index. */
		const Index& getIndex() const;

		/** On gaussian smoothing changed. */
		void onGaussianSmoothingChanged(wxCommandEvent &event);

		/** On min changed. */
		void onMinChanged(wxCommandEvent &event);

		/** On max changed. */
		void onMaxChanged(wxCommandEvent &event);

/*		enum class ID {
			GaussianSmoothing,
			Min,
			Max,
			Index
		};*/
	protected:
		DECLARE_EVENT_TABLE();
	private:
		/** Gaussian smoothing label. */
		wxStaticText gaussianSmoothingLabel;

		/** Text box for smoothing parameter. */
		wxTextCtrl gaussianSmoothingTextBox;

		/** Gaussian smoothing. */
		double gaussianSmoothing;

		/** Gaussian smoothing label. */
		wxStaticText minLabel;

		/** Text box for min. */
		wxTextCtrl minTextBox;

		/** Min. */
		double min;

		/** Gaussian smoothing label. */
		wxStaticText maxLabel;

		/** Text box for max. */
		wxTextCtrl maxTextBox;

		/** Max. */
		double max;

		/** Index panel. */
		IndexPanel indexPanel;
	};

	/** Control panel. */
	ControlPanel controlPanel;

	static const wxWindowID GAUSSIAN_SMOOTHING_ID;
	static const wxWindowID MIN_ID;
	static const wxWindowID MAX_ID;

	/** Update plot. */
	void updatePlot();
};

inline double LDOSPanel::ControlPanel::getGaussianSmoothing() const{
	return gaussianSmoothing;
}

inline double LDOSPanel::ControlPanel::getMin() const{
	return min;
}

inline double LDOSPanel::ControlPanel::getMax() const{
	return max;
}

inline const Index& LDOSPanel::ControlPanel::getIndex() const{
	return indexPanel.getIndex();
}

};	//End namespace TBTK

#endif
/// @endcond
