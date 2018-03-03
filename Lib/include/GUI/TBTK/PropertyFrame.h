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
 *  @file PropertyFrame.h
 *  @brief Frame for displaying properties.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_FRAME
#define COM_DAFER45_TBTK_PROPERTY_FRAME

#include "TBTK/ImagePanel.h"
#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/Property/DOS.h"
#include "TBTK/Property/LDOS.h"

#include <wx/wx.h>
#include <wx/sizer.h>

namespace TBTK{

class PropertyFrame : public wxFrame{
public:
	/** Constructor. */
	PropertyFrame(wxWindow *parent);

	/** Destructor. */
	~PropertyFrame();

	/** Set DOS as property. */
	void setProperty(const Property::DOS &dos);

	/** Set DOS as property. */
	void setProperty(const Property::LDOS &ldos);

	/** On paint event. */
	void onPaintEvent(wxPaintEvent &event);

	/** On size event. */
	void onSizeEvent(wxSizeEvent &event);

	/** On menu->file->open. */
	void onMenuFileOpen(wxCommandEvent &event);

	/** On menu->file->quit. */
	void onMenuFileQuit(wxCommandEvent &event);

	/** On menu->help->about. */
	void onMenuHelpAbout(wxCommandEvent &event);
protected:
	DECLARE_EVENT_TABLE();
private:
	/** Pointer to the property. */
//	void *property;

	/** Current panel. */
	wxPanel *currentPanel;

	/** Enum class used to indicate what property the property pointer
	 *  points to. */
	enum class PropertyType {
		None,
		DOS,
		LDOS
	};

	/** Set property panel. */
	void setPropertyPanel(PropertyType propertyType);

	/** Current property type. */
	PropertyType currentPropertyType;
};

};	//End namespace TBTK

#endif
