const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/jpg']
const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']
const MAX_FILE_SIZE_MB = 8
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

export function validateImageFile(file) {
  if (!file) {
    return { valid: false, error: 'Please choose an X-ray image first.' }
  }

  const extension = file.name.split('.').pop()?.toLowerCase()

  if (!extension || !ALLOWED_EXTENSIONS.includes(extension)) {
    return {
      valid: false,
      error: 'Unsupported file extension. Use JPG, JPEG, or PNG.',
    }
  }

  if (!ALLOWED_MIME_TYPES.includes(file.type)) {
    return {
      valid: false,
      error: 'Unsupported image format. Please upload JPG, JPEG, or PNG.',
    }
  }

  if (file.size > MAX_FILE_SIZE_BYTES) {
    return {
      valid: false,
      error: `File is too large. Maximum size is ${MAX_FILE_SIZE_MB} MB.`,
    }
  }

  return { valid: true }
}

export function formatFileSize(sizeBytes) {
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`
  }

  if (sizeBytes < 1024 * 1024) {
    return `${(sizeBytes / 1024).toFixed(1)} KB`
  }

  return `${(sizeBytes / (1024 * 1024)).toFixed(1)} MB`
}
