"""
Tests for PDF combination functionality in predictive_checks.py.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest
from beartype.typing import List

from pyrovelocity.plots.predictive import combine_pdfs


class TestPDFCombination:
    """Test PDF combination functionality."""

    def create_dummy_pdf(self, filepath: Path) -> None:
        """Create a dummy PDF file for testing."""
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, f"Test PDF: {filepath.name}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"Dummy PDF: {filepath.stem}")
        fig.savefig(filepath, format='pdf', bbox_inches='tight')
        plt.close(fig)

    def test_combine_pdfs_basic(self):
        """Test basic PDF combination functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create some dummy PDF files
            pdf_files = [
                "prior_parameter_marginals.pdf",
                "prior_parameter_relationships.pdf", 
                "prior_expression_validation.pdf"
            ]
            
            for pdf_file in pdf_files:
                self.create_dummy_pdf(temp_path / pdf_file)
            
            # Combine PDFs
            output_file = "combined_test.pdf"
            combine_pdfs(
                pdf_directory=str(temp_path),
                output_filename=output_file,
                pdf_pattern="*.pdf"
            )
            
            # Check that combined file was created
            combined_path = temp_path / output_file
            assert combined_path.exists(), "Combined PDF file should be created"
            assert combined_path.stat().st_size > 0, "Combined PDF should not be empty"

    def test_combine_pdfs_with_exclusions(self):
        """Test PDF combination with exclusion patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create PDF files including some to exclude
            pdf_files = [
                "prior_parameter_marginals.pdf",
                "prior_expression_validation.pdf",
                "combined_old.pdf",  # Should be excluded
                "temp_file.pdf"
            ]
            
            for pdf_file in pdf_files:
                self.create_dummy_pdf(temp_path / pdf_file)
            
            # Combine PDFs with exclusions
            output_file = "combined_new.pdf"
            combine_pdfs(
                pdf_directory=str(temp_path),
                output_filename=output_file,
                pdf_pattern="*.pdf",
                exclude_patterns=["combined_*.pdf", "temp_*.pdf"]
            )
            
            # Check that combined file was created
            combined_path = temp_path / output_file
            assert combined_path.exists(), "Combined PDF file should be created"

    def test_combine_pdfs_pattern_matching(self):
        """Test PDF combination with specific patterns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create PDF files with different patterns
            pdf_files = [
                "prior_parameter_marginals.pdf",
                "prior_parameter_relationships.pdf",
                "prior_expression_validation.pdf",
                "posterior_parameter_marginals.pdf"
            ]
            
            for pdf_file in pdf_files:
                self.create_dummy_pdf(temp_path / pdf_file)
            
            # Combine only prior parameter PDFs
            output_file = "prior_parameters_only.pdf"
            combine_pdfs(
                pdf_directory=str(temp_path),
                output_filename=output_file,
                pdf_pattern="prior_parameter*.pdf"
            )
            
            # Check that combined file was created
            combined_path = temp_path / output_file
            assert combined_path.exists(), "Combined PDF file should be created"

    def test_combine_pdfs_empty_directory(self):
        """Test behavior when no PDFs are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Try to combine PDFs from empty directory
            output_file = "combined_empty.pdf"
            
            # Should not raise an error, just print a message
            combine_pdfs(
                pdf_directory=str(temp_path),
                output_filename=output_file,
                pdf_pattern="*.pdf"
            )
            
            # Combined file should not be created
            combined_path = temp_path / output_file
            assert not combined_path.exists(), "No combined PDF should be created for empty directory"

    def test_combine_pdfs_nonexistent_directory(self):
        """Test behavior when directory doesn't exist."""
        with pytest.raises(FileNotFoundError):
            combine_pdfs(
                pdf_directory="/nonexistent/directory",
                output_filename="test.pdf"
            )

    @patch('pypdf.PdfReader')
    def test_combine_pdfs_corrupted_file(self, mock_pdf_reader):
        """Test behavior when a PDF file is corrupted."""
        # Mock PdfReader to raise an exception
        mock_pdf_reader.side_effect = Exception("Corrupted PDF")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create a dummy PDF file
            self.create_dummy_pdf(temp_path / "test.pdf")

            # Should handle the exception gracefully
            output_file = "combined_with_error.pdf"
            combine_pdfs(
                pdf_directory=str(temp_path),
                output_filename=output_file,
                pdf_pattern="*.pdf"
            )

            # Combined file should still be created (empty)
            combined_path = temp_path / output_file
            assert combined_path.exists(), "Combined PDF should be created even with errors"

    def test_combine_pdfs_sorting(self):
        """Test that PDF files are combined in sorted order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create PDF files with names that should be sorted
            pdf_files = [
                "c_third.pdf",
                "a_first.pdf", 
                "b_second.pdf"
            ]
            
            for pdf_file in pdf_files:
                self.create_dummy_pdf(temp_path / pdf_file)
            
            # Combine PDFs
            output_file = "combined_sorted.pdf"
            combine_pdfs(
                pdf_directory=str(temp_path),
                output_filename=output_file,
                pdf_pattern="*.pdf"
            )
            
            # Check that combined file was created
            combined_path = temp_path / output_file
            assert combined_path.exists(), "Combined PDF file should be created"
            
            # Note: We can't easily test the internal order without 
            # reading the PDF content, but the function should sort the files


if __name__ == "__main__":
    pytest.main([__file__])
