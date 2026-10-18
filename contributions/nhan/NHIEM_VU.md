# Chu Đức Thành Nhân — Demo, tái lập và tích hợp báo cáo

Trạng thái: **được phân công, chưa xác nhận hoàn thành**.

Nhánh: `work/nhan`. Kiểm tra chéo: Lê Bá Khánh Bình.

## Các việc tự thực hiện

1. Cài đặt môi trường sạch theo README sau khi ghép đủ bốn phần; kiểm tra pip install -e .
2. Chạy toàn bộ unittest, notebook và demo; đối chiếu score/threshold với model đã khóa.
3. Tích hợp nhận xét báo cáo của cả nhóm; chỉ cập nhật đóng góp đã có minh chứng và metadata do nhóm xác nhận.
4. Xuất PDF, kiểm tra bố cục, chạy check_submission.py rồi tạo lại và xác minh SHA256SUMS.
5. Kiểm tra chéo việc nghiệm thu dữ liệu của Bình và tạo gói nộp sau khi các PR đã được merge.

## File nhập vào repo

Copy **nội dung bên trong** ZIP cá nhân vào gốc repo; giữ nguyên đường dẫn. Các file kết quả/log là đầu ra đã có trước khi tạo repo, không phải minh chứng lần chạy của cá nhân.

- `.gitignore`
- `CONTRIBUTIONS.md`
- `demo/app.py`
- `demo/index.html`
- `evidence/delivery_checks.json`
- `evidence/demo_examples.json`
- `evidence/demo_mobile.png`
- `evidence/demo_screenshot.png`
- `evidence/demo_smoke_2026-10-15.json`
- `evidence/demo_smoke_test.json`
- `evidence/install_package_2026-10-15.log`
- `evidence/install_requirements_2026-10-15.log`
- `evidence/notebook_execution_2026-10-15.json`
- `evidence/notebook_execution_2026-10-18.json`
- `evidence/reproduction_check.json`
- `evidence/reproduction_full.log`
- `evidence/revision_notes_2026-10-15.md`
- `evidence/revision_notes_2026-10-18.md`
- `evidence/unit_tests.log`
- `evidence/unit_tests_2026-10-13.log`
- `evidence/unit_tests_2026-10-15.log`
- `evidence/unit_tests_2026-10-18.log`
- `Final_CRISP_DM.ipynb`
- `PEER_ASSESSMENT_MAU_RIENG.md`
- `pyproject.toml`
- `README.md`
- `report/Bao_cao_Cuoi_ky_Nhom7_CRISP_DM.docx`
- `report/Bao_cao_Cuoi_ky_Nhom7_CRISP_DM.pdf`
- `report/export_pdf.ps1`
- `requirements-notebook.txt`
- `requirements.txt`
- `scripts/build_notebook.py`
- `scripts/check_submission.py`
- `scripts/make_sha256sums.py`
- `scripts/reproduce.py`
- `scripts/verify_bundle.py`
- `src/fakejobs/__init__.py`
- `src/fakejobs/inference.py`
- `SHA256SUMS.txt`

## Nhật ký nghiệm thu cần tự điền

| Ngày thực tế | Việc đã tự làm / lệnh chạy | Kết quả và hạn chế | File/log/commit/PR | Người kiểm tra chéo |
|---|---|---|---|---|
| Chưa ghi nhận | | | | |

Lưu log mới tại `contributions/nhan/evidence/` khi đã chạy; không chép log lịch sử rồi đổi tên thành log cá nhân. Không ghi đã viết hoặc chạy lại nếu chỉ nhập file.
