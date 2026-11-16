import { Component, type ErrorInfo, type ReactNode } from 'react';
import { Notice } from '../ui/Notice';
import { Card } from '../ui/Card';

interface Props { children: ReactNode; resetKey?: string }
interface State { error: Error | null }

/** A render crash in one view shows an error card instead of a blank page; changing route resets it. */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };
  static getDerivedStateFromError(error: Error): State { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('View crashed', error, info.componentStack); }
  componentDidUpdate(prev: Props) { if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null }); }
  render() {
    if (this.state.error) {
      return <Card><Notice tone="danger">Failed to load module: {this.state.error.message}</Notice></Card>;
    }
    return this.props.children;
  }
}
