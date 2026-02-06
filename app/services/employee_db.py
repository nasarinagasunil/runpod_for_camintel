"""
Minimal EmployeeDatabase for RunPod ML Service.
Uses pickle file for embeddings instead of PostgreSQL.

This provides the same interface as backend/app/services/employee_db.py
but stores data in a local pickle file (can be mounted or pre-loaded).
"""
import pickle
import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class EmployeeDatabase:
    """
    Simple file-based employee embeddings storage for RunPod.
    Provides same interface as the PostgreSQL-backed version.
    """
    
    def __init__(self, embeddings_path: str = "/app/employee_embeddings.pkl"):
        self.embeddings_path = embeddings_path
        self.employees: Dict = {}
        self.rfid_to_employee: Dict = {}
        
        self.load_database()
    
    def load_database(self):
        """Load embeddings from pickle file."""
        if os.path.exists(self.embeddings_path):
            try:
                with open(self.embeddings_path, 'rb') as f:
                    data = pickle.load(f)
                    
                if isinstance(data, dict):
                    self.employees = data
                    
                    # Build RFID index
                    for emp_id, emp_data in self.employees.items():
                        rfid = emp_data.get('rfid')
                        if rfid:
                            self.rfid_to_employee[rfid] = emp_id
                            
                logger.info(f"Loaded {len(self.employees)} employees from {self.embeddings_path}")
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
        else:
            logger.warning(f"No embeddings file found at {self.embeddings_path}")
    
    def enroll(self, emp_id: str, name: str, role: str, 
               embeddings: List[np.ndarray], rfid: str = None):
        """Add or update employee. Saves to pickle file."""
        self.employees[emp_id] = {
            "name": name,
            "role": role,
            "embeddings": embeddings,
            "rfid": rfid
        }
        
        if rfid:
            self.rfid_to_employee[rfid] = emp_id
        
        self._save()
        logger.info(f"Enrolled {name} ({emp_id})")
    
    def _save(self):
        """Save to pickle file."""
        try:
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(self.employees, f)
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Get all embeddings as matrix for similarity search."""
        all_emb, emp_ids = [], []
        
        for emp_id, data in self.employees.items():
            for emb in data.get('embeddings', []):
                if isinstance(emb, np.ndarray) and emb.size > 0:
                    all_emb.append(emb)
                    emp_ids.append(emp_id)
        
        if not all_emb:
            return np.array([]), []
        
        return np.vstack(all_emb), emp_ids
    
    def get_by_rfid(self, rfid: str) -> Optional[Dict]:
        """Look up employee by RFID tag."""
        if not rfid:
            return None
            
        emp_id = self.rfid_to_employee.get(rfid)
        if emp_id:
            return {"employee_id": emp_id, **self.employees[emp_id]}
        return None
    
    def delete_employee(self, emp_id: str) -> bool:
        """Remove employee from database."""
        if emp_id in self.employees:
            rfid = self.employees[emp_id].get('rfid')
            if rfid:
                self.rfid_to_employee.pop(rfid, None)
            del self.employees[emp_id]
            self._save()
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        return {
            "total_employees": len(self.employees),
            "backend": "pickle_file",
            "file_path": self.embeddings_path
        }
