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
 *  @file DOSPanel.h
 *  @brief Panel for DOS.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_DOS_PANEL
#define COM_DAFER45_TBTK_DOS_PANEL

#include "TBTK/ImagePanel.h"
#include "TBTK/Property/DOS.h"

#include <wx/wx.h>
#include <wx/sizer.h>

namespace TBTK{

class DOSPanel : public wxPanel{
public:
	/** Constructor. */
	DOSPanel(wxWindow *parent);

	/** Destructor. */
	~DOSPanel();

	/** Set DOS. */
	void setDOS(const Property::DOS &dos);

	/** On paint event. */
	void onPaintEvent(wxPaintEvent &event);

	/** On size event. */
	void onSizeEvent(wxSizeEvent &event);

	/** On change. */
	void onChangeEvent(wxCommandEvent &event);

	/** Redraw. */
//	void redraw();
protected:
	DECLARE_EVENT_TABLE();
private:
	/** Pointer to the property. */
	Property::DOS *dos;

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

		/** On gaussian smoothing changed. */
		void onGaussianSmoothingChanged(wxCommandEvent &event);

		/** On min changed. */
		void onMinChanged(wxCommandEvent &event);

		/** On max changed. */
		void onMaxChanged(wxCommandEvent &event);

/*		enum ID {
			GaussianSmoothing,
			Min,
			Max
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
	};

	/** Control panel. */
	ControlPanel controlPanel;

	static const wxWindowID GAUSSIAN_SMOOTHING_ID;
	static const wxWindowID MIN_ID;
	static const wxWindowID MAX_ID;

	/** Update plot. */
	void updatePlot();
};

inline double DOSPanel::ControlPanel::getGaussianSmoothing() const{
	return gaussianSmoothing;
}

inline double DOSPanel::ControlPanel::getMin() const{
	return min;
}

inline double DOSPanel::ControlPanel::getMax() const{
	return max;
}

};	//End namespace TBTK

#endif
