/* 
#  -------------------------------------------------------------------------
#  pyCGNS.WRA - Python package for CFD General Notation System - WRAper
#  See license.txt file in the root directory of this Python module source  
#  -------------------------------------------------------------------------
#  $Release:  $
#  ------------------------------------------------------------------------- 
*/
#ifdef __ADF_IN_SOURCES__
#include "adfh/ADFH.h"
#include "adf/ADF.h"
#else
#include "ADFH.h"
#include "ADF.h"
#endif

/* risk here, the struct may be wrong... */

/* from cgns_header.h */
typedef char char_33[33];
typedef char const cchar_33[33];
typedef int int_6[6];
typedef int int_3[3];

#ifndef __USE_NUMARRAY__
#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

/* file types */

#define CG_FILE_NONE 0
#define CG_FILE_ADF  1
#define CG_FILE_HDF5 2
#define CG_FILE_XML  3

#define CG_OK	     0
#define STORAGE_ADF      CG_FILE_ADF 
#define STORAGE_HDF      CG_FILE_HDF5 
#define STORAGE_UNKNOWN  CG_FILE_NONE

typedef struct {
        char *filename;         /* name of file                         */
	int version;		/* version of the CGNS file * 1000	*/
	double rootid;
        int mode;               /* reading or writing                   */
        int file_number;        /* external identifier                  */
        int deleted;            /* number of deleted nodes              */
        int added;              /* number of added nodes                */
	char_33 dtb_version;	/* ADF Database Version			*/
	char_33 creation_date;	/* creation date of the file		*/
	char_33 modify_date;	/* last modification date of the file	*/
	char_33 adf_lib_version;/* ADF Library Version			*/
        int nbases;             /* number of bases in the file          */

        /* pyCGNS -> FORCE CAST TO VOID * */
        void *base;       	/* ptrs to in-memory copies of bases    */
} cgns_file;

extern cgns_file *cgns_files;
extern cgns_file *cg;
extern int n_cgns_files;
extern void *posit;
extern char_33 posit_label;
extern int posit_base, posit_zone, temp_index;
extern int CGNSLibVersion;	/* CGNSLib Version number		*/

cgns_file    *cgi_get_file   (int file_number);
void *cgi_get_posit(int fn, int B, int n, int *index, char **label, int *ier);

/* options flags */

#define __CGNS__DEFAULT__FLAGS__       0x0001

#define __CGNS__NO__FLAGS__            0x0000
#define __CGNS__FORCE_FORTRAN_INDEX__  0x0001
#define __CGNS__FORCE_PATH__SEARCH__   0x0002

#define __CGNS__HAS__ADF__ 1

#ifdef __CGNS__HAS__ADF__
/* macro to easily switch on two storage types */
/* we're lucky there is no return value...     */
#define ADF__Children_Names(PID, istart, ilen, name_length, ilen_ret, names, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Children_Names(PID, istart, ilen, name_length, ilen_ret, names, error_return ) ; } else {ADFH_Children_Names(PID, istart, ilen, name_length, ilen_ret, names, error_return ) ;}
#define ADF__Children_IDs(PID, istart, ilen, ilen_ret, IDs, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Children_IDs(PID, istart, ilen, ilen_ret, IDs, error_return ) ; } else {ADFH_Children_IDs(PID, istart, ilen, ilen_ret, IDs, error_return ) ;}
#define ADF__Create(PID, name, ID, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Create(PID, name, ID, error_return ) ; } else {ADFH_Create(PID, name, ID, error_return ) ;}
#define ADF__Database_Close(ID, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Database_Close(ID, error_return ) ; } else {ADFH_Database_Close(ID, error_return ) ;}
#define ADF__Database_Delete(filename, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Database_Delete(filename, error_return ) ; } else {ADFH_Database_Delete(filename, error_return ) ;}
#define ADF__Database_Garbage_Collection(ID, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Database_Garbage_Collection(ID, error_return ) ; } else {ADFH_Database_Garbage_Collection(ID, error_return ) ;}
#define ADF__Database_Get_Format(Root_ID, format, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Database_Get_Format(Root_ID, format, error_return ) ; } else {ADFH_Database_Get_Format(Root_ID, format, error_return ) ;}
#define ADF__Database_Open(filename, status, format, root_ID, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Database_Open(filename, status, format, root_ID, error_return ) ; } else {ADFH_Database_Open(filename, status, format, root_ID, error_return ) ;}
#define ADF__Database_Valid(filename, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Database_Valid(filename, error_return ) ; } else {ADFH_Database_Valid(filename, error_return ) ;}
#define ADF__Database_Set_Format(Root_ID, format, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Database_Set_Format(Root_ID, format, error_return ) ; } else {ADFH_Database_Set_Format(Root_ID, format, error_return ) ;}
#define ADF__Database_Version(Root_ID, version, creation_date, modification_date, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Database_Version(Root_ID, version, creation_date, modification_date, error_return ) ; } else {ADFH_Database_Version(Root_ID, version, creation_date, modification_date, error_return ) ;}
#define ADF__Delete(PID, ID, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Delete(PID, ID, error_return ) ; } else {ADFH_Delete(PID, ID, error_return ) ;}
#define ADF__Error_Message(error_return_input, error_string ) if (self->storagetype==STORAGE_ADF){ADF_Error_Message(error_return_input, error_string ) ; } else {ADFH_Error_Message(error_return_input, error_string ) ;}
#define ADF__Flush_to_Disk(ID, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Flush_to_Disk(ID, error_return ) ; } else {ADFH_Flush_to_Disk(ID, error_return ) ;}
#define ADF__Get_Data_Type(ID, data_type, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Get_Data_Type(ID, data_type, error_return ) ; } else {ADFH_Get_Data_Type(ID, data_type, error_return ) ;}
#define ADF__Get_Dimension_Values(ID, dim_vals, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Get_Dimension_Values(ID, dim_vals, error_return ) ; } else {ADFH_Get_Dimension_Values(ID, dim_vals, error_return ) ;}
#define ADF__Get_Error_State(error_state, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Get_Error_State(error_state, error_return ) ; } else {ADFH_Get_Error_State(error_state, error_return ) ;}
#define ADF__Get_Label(ID, label, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Get_Label(ID, label, error_return ) ; } else {ADFH_Get_Label(ID, label, error_return ) ;}
#define ADF__Get_Link_Path(ID, filename, link_path, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Get_Link_Path(ID, filename, link_path, error_return ) ; } else {ADFH_Get_Link_Path(ID, filename, link_path, error_return ) ;}
#define ADF__Get_Name(ID, name, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Get_Name(ID, name, error_return ) ; } else {ADFH_Get_Name(ID, name, error_return ) ;}
#define ADF__Get_Node_ID(PID, name, ID, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Get_Node_ID(PID, name, ID, error_return ) ; } else {ADFH_Get_Node_ID(PID, name, ID, error_return ) ;}
#define ADF__Get_Number_of_Dimensions(ID, num_dims, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Get_Number_of_Dimensions(ID, num_dims, error_return ) ; } else {ADFH_Get_Number_of_Dimensions(ID, num_dims, error_return ) ;}
#define ADF__Get_Root_ID(ID, Root_ID, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Get_Root_ID(ID, Root_ID, error_return ) ; } else {ADFH_Get_Root_ID(ID, Root_ID, error_return ) ;}
#define ADF__Is_Link(ID, link_path_length, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Is_Link(ID, link_path_length, error_return ) ; } else {ADFH_Is_Link(ID, link_path_length, error_return ) ;}
#define ADF__Library_Version(version, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Library_Version(version, error_return ) ; } else {ADFH_Library_Version(version, error_return ) ;}
#define ADF__Link(PID, name, file, name_in_file, ID, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Link(PID, name, file, name_in_file, ID, error_return ) ; } else {ADFH_Link(PID, name, file, name_in_file, ID, error_return ) ;}
#define ADF__Link_Size(ID, file_length, name_length, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Link_Size(ID, file_length, name_length, error_return ) ; } else {ADFH_Link_Size(ID, file_length, name_length, error_return ) ;}
#define ADF__Move_Child(PID, ID, NPID, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Move_Child(PID, ID, NPID, error_return ) ; } else {ADFH_Move_Child(PID, ID, NPID, error_return ) ;}
#define ADF__Number_of_Children(ID, num_children, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Number_of_Children(ID, num_children, error_return ) ; } else {ADFH_Number_of_Children(ID, num_children, error_return ) ;}
#define ADF__Put_Dimension_Information(ID, data_type, dims, dim_vals, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Put_Dimension_Information(ID, data_type, dims, dim_vals, error_return ) ; } else {ADFH_Put_Dimension_Information(ID, data_type, dims, dim_vals, error_return ) ;}
#define ADF__Put_Name(PID, ID, name, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Put_Name(PID, ID, name, error_return ) ; } else {ADFH_Put_Name(PID, ID, name, error_return ) ;}
#define ADF__Read_All_Data(ID, data, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Read_All_Data(ID, data, error_return ) ; } else {ADFH_Read_All_Data(ID, data, error_return ) ;}
#define ADF__Read_Block_Data(ID, b_start, b_end, data, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Read_Block_Data(ID, b_start, b_end, data, error_return ) ; } else {ADFH_Read_Block_Data(ID, b_start, b_end, data, error_return ) ;}
#define ADF__Read_Data(ID, s_start, s_end, s_stride, m_num_dims, m_dims, m_start, m_end, m_stride, data, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Read_Data(ID, s_start, s_end, s_stride, m_num_dims, m_dims, m_start, m_end, m_stride, data, error_return ) ; } else {ADFH_Read_Data(ID, s_start, s_end, s_stride, m_num_dims, m_dims, m_start, m_end, m_stride, data, error_return ) ;}
#define ADF__Set_Error_State(error_state, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Set_Error_State(error_state, error_return ) ; } else {ADFH_Set_Error_State(error_state, error_return ) ; }
#define ADF__Set_Label(ID, label, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Set_Label(ID, label, error_return ) ; } else {ADFH_Set_Label(ID, label, error_return ) ;}
#define ADF__Write_All_Data(ID, data, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Write_All_Data(ID, data, error_return ) ; } else {ADFH_Write_All_Data(ID, data, error_return ) ;}
#define ADF__Write_Block_Data(ID, b_start, b_end, data, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Write_Block_Data(ID, b_start, b_end, data, error_return ) ; } else {ADFH_Write_Block_Data(ID, b_start, b_end, data, error_return ) ;}
#define ADF__Write_Data(ID, s_start, s_end, s_stride, m_num_dims, m_dims, m_start, m_end, m_stride, data, error_return ) if (self->storagetype==STORAGE_ADF){ADF_Write_Data(ID, s_start, s_end, s_stride, m_num_dims, m_dims, m_start, m_end, m_stride, data, error_return ) ; } else {ADFH_Write_Data(ID, s_start, s_end, s_stride, m_num_dims, m_dims, m_start, m_end, m_stride, data, error_return ) ;}
#define ADF__Release_ID (   ID )if (self->storagetype==STORAGE_ADF){ADF_Release_ID (   ID ); } else {ADFH_Release_ID (   ID );}

#else

#define ADF__Children_Names ADFH_Children_Names
#define ADF__Children_IDs ADFH_Children_IDs
#define ADF__Create ADFH_Create
#define ADF__Database_Close ADFH_Database_Close
#define ADF__Database_Delete ADFH_Database_Delete
#define ADF__Database_Garbage_Collection ADFH_Database_Garbage_Collection
#define ADF__Database_Get_Format ADFH_Database_Get_Format
#define ADF__Database_Open ADFH_Database_Open
#define ADF__Database_Valid ADFH_Database_Valid
#define ADF__Database_Set_Format ADFH_Database_Set_Format
#define ADF__Database_Version ADFH_Database_Version
#define ADF__Delete ADFH_Delete
#define ADF__Error_Message ADFH_Error_Message
#define ADF__Flush_to_Disk ADFH_Flush_to_Disk
#define ADF__Get_Data_Type ADFH_Get_Data_Type
#define ADF__Get_Dimension_Values ADFH_Get_Dimension_Values
#define ADF__Get_Error_State ADFH_Get_Error_State
#define ADF__Get_Label ADFH_Get_Label
#define ADF__Get_Link_Path ADFH_Get_Link_Path
#define ADF__Get_Name ADFH_Get_Name
#define ADF__Get_Node_ID ADFH_Get_Node_ID
#define ADF__Get_Number_of_Dimensions ADFH_Get_Number_of_Dimensions
#define ADF__Get_Root_ID ADFH_Get_Root_ID
#define ADF__Is_Link ADFH_Is_Link
#define ADF__Library_Version ADFH_Library_Version
#define ADF__Link ADFH_Link
#define ADF__Link_Size ADFH_Link_Size
#define ADF__Move_Child ADFH_Move_Child
#define ADF__Number_of_Children ADFH_Number_of_Children
#define ADF__Put_Dimension_Information ADFH_Put_Dimension_Information
#define ADF__Put_Name ADFH_Put_Name
#define ADF__Read_All_Data ADFH_Read_All_Data
#define ADF__Read_Block_Data ADFH_Read_Block_Data
#define ADF__Read_Data ADFH_Read_Data
#define ADF__Set_Error_State ADFH_Set_Error_State
#define ADF__Set_Label ADFH_Set_Label
#define ADF__Write_All_Data ADFH_Write_All_Data
#define ADF__Write_Block_Data ADFH_Write_Block_Data
#define ADF__Write_Data ADFH_Write_Data
#define ADF__Release_ID ADFH_Release_ID

#endif
